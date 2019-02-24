#U-Net for semantically segmenting roads in aerial imagery.
#Uses all 5 possible input channels
#Based loosely on sample code from https://github.com/mrgloom/keras-semantic-segmentation-example/blob/master/binary_segmentation/binary_crossentropy_example.py
#and https://github.com/zhixuhao/unet

"""
Instructions:

    - There are two modes of use- training the unet based on the images in the roads/training folder, or generating predicted masks.
    - Use command line arguments to choose which mode- the -m parameter can be either 'train' or 'predict'.
    - When training is complete, it will automatically show the prediction for the first training image.
    - If predicting, you can run it on images from the ./training or ./unseen folders. Use command line args to choose which.

Notes:

    - The generator functions assume the number of images is an integer multiple of the batch size
        and it will crash if this isn't the case due to partial batches
    - The roads/unseen folder is not meant for validation, just testing on arbitrary images 
"""
from __future__ import print_function

import os
import cv2
import sys
import util
import time
import keras
import scipy
import psutil
import models
import skimage
import warnings
import argparse
import feature_vis
import numpy as np
import random as rn
from skimage import measure
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers import concatenate
import skimage.morphology as morphology
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.layers.core import Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from util import get_single_image, number_of_files, concatenate_5_layers, get_5layer_img, load_data
from util import get_dxy_from_txt, get_model

#Parameters
INPUT_CHANNELS = 5 #RGB + XY derivatives
NUMBER_OF_CLASSES = util.NUMBER_OF_CHANNELS #Number of mask layers to output
IMAGE_W = util.IMAGE_W
IMAGE_H = util.IMAGE_H
batch_size = util.BATCH_SIZE #Number of images fed to the network at a time
TRAINING_DATA_DIR = util.DATA_SOURCE
SAVED_MODEL = "./past_models/5layer_400x28.h5"


"""Loads images from computer, makes them into input batches"""
def batch_generator(batch_size):
    starting_point = 0
    print("Loading files using batch_generator")

    while True:
        image_list = load_data(TRAINING_DATA_DIR+"rgb", batch_size, starting_point) #Get various images out of folders
        dxy_list = util.batch_from_txt(TRAINING_DATA_DIR+"txt", batch_size, starting_point)
        mask_list = load_data(TRAINING_DATA_DIR+"mask", batch_size, starting_point)

        with warnings.catch_warnings(): #Rescale, resize, etc
            warnings.simplefilter("ignore")
            image_list = [skimage.transform.resize(image, (IMAGE_H, IMAGE_W)) for image in image_list]
            mask_list = [skimage.transform.resize(mask, (IMAGE_H, IMAGE_W)) for mask in mask_list]
            mask_list = [skimage.color.rgb2gray(i) for i in mask_list]
            mask_list = [skimage.exposure.rescale_intensity(i) for i in mask_list]

        image_list = np.array(image_list, dtype=np.float32)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list = np.nan_to_num(mask_list)
        mask_list = mask_list.reshape(batch_size, IMAGE_H*IMAGE_W, NUMBER_OF_CLASSES)
        
        image_list = np.concatenate((image_list, dxy_list), 3) #Combine RGB and XY channels into 5 layer image tiles

        starting_point += 8 #Move index along to grab next batch
        if(starting_point >= number_of_files()):
            starting_point = 0

        yield image_list, mask_list




"""Generates augmented 5 layer image samples"""
def from_txt_generator(batch_size): 
    print("Getting augmented data from text files")
    index = rn.randint(0,27)*8 #Start at random image\

    img_datagen = idg( #Transformation parameters
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True,
        fill_mode="reflect"
        )

    rgb_datagen = idg(
        rescale=1./255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True,
        fill_mode="reflect"
    )
    
    while(1):
        seed = rn.randint(0,5000)
        img_list = load_data(TRAINING_DATA_DIR+"rgb", batch_size, index) #Get images
        dxy_list = util.batch_from_txt(TRAINING_DATA_DIR+"txt", batch_size, index, False) 
        mask_list = load_data(TRAINING_DATA_DIR+"mask", batch_size, index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_list = [skimage.transform.resize(mask, (IMAGE_H, IMAGE_W)) for mask in mask_list]
            mask_list = [skimage.color.rgb2gray(i) for i in mask_list]
            mask_list = [skimage.exposure.rescale_intensity(i) for i in mask_list]

        img_list = np.array(img_list, dtype=np.float32)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list = np.expand_dims(mask_list, 3)
        mask_list = np.nan_to_num(mask_list) #Need this for some reason???

        index += batch_size
        if(index >= number_of_files()):
            index = 0

        #Generate next batch of images by applying random transformations
        img_gen = rgb_datagen.flow(img_list, mask_list, batch_size=batch_size, shuffle=True, seed=seed)    
        dx_gen = img_datagen.flow(dxy_list, mask_list, batch_size=batch_size, shuffle=True, seed=seed)
        mask_gen = img_datagen.flow(mask_list, mask_list, batch_size=batch_size, shuffle=True, seed=seed)
    
        img = img_gen.next()[0] #flow method always produces a tuple of data+labels, but we only want the data
        dxy = dx_gen.next()[0]
        mask = mask_gen.next()[0]

        out = np.concatenate((img, dxy), 3) #Combine layers
        mask = mask.reshape(batch_size, IMAGE_H*IMAGE_W, NUMBER_OF_CLASSES)

        yield out, mask





"""Generates predicted mask for a 5 layer input image"""
def get_prediction(just_trained, index, parent_dir=TRAINING_DATA_DIR):
    print("Generating predicted mask")
    if (just_trained): #Means we have just completed a training run, so use the most recent model
        print("Getting most recent model")
        model = load_model('5layer.h5')
    else:
        model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL)

    
    img = util.get_5layer_img(index, parent_dir)
    util.show_layers(img)

    y_pred = model.predict(img[None,...].astype(np.float32))[0]    
    y_pred = y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))

    return y_pred



  


"""Shows predicted mask for inspection"""
def visually_inspect_result(just_trained=False, unseen=False, index=0, parent_dir=TRAINING_DATA_DIR):
    print("Getting prediction for visual inspection")

    rgb = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32) #Image to display
    y_pred = get_prediction(just_trained, index, parent_dir)
    #rgb[:,:,2] = rgb[:,:,2]+np.squeeze(y_pred)
 
    #Show ground truth mask (if we are using the training set for our sample prediction)
    if (unseen == False):
        mask = get_single_image(parent_dir, "mask", index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = skimage.color.rgb2gray(mask)
            mask = skimage.exposure.rescale_intensity(mask)
            mask = np.nan_to_num(mask)
        cv2.imshow('ground truth',mask)

        y_pred2 = np.copy(y_pred)
        iou = util.intersection_over_union(y_pred2, mask)
        print("IoU = %f" % iou)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('input image',rgb)
    cv2.imshow('5layer result',y_pred)
    
    print("Displaying images")
    cv2.waitKey(0)
    return









"""Trains the unet model using the data generators"""
def train(steps=28, epochs=1, unet=0):
    model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL, unet)

    callbacks = [
        ProgbarLogger(count_mode='steps', stateful_metrics=None),
        ModelCheckpoint('5l_weights_last.h5', monitor='val_loss', save_best_only=False, verbose=0), 
        util.MemLeakCallback()
    ]


    print("Training network")
    history = model.fit_generator(
        generator=from_txt_generator(batch_size),
        steps_per_epoch=steps, #Number of samples obtained using batch_gen or variation_gen (28 for whole dataset)
        validation_data=batch_generator(batch_size),
        epochs=epochs, #Number of passes over whole dataset
        validation_steps=1,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    model.save("5layer.h5")
    return 



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='predict', required=True, type=str, help="[train | predict]: Whether to train the network or make predictions on images")
    parser.add_argument("-v", "--unet-version", default=0, type=int, help="[1 <= int <= 4]: (train mode only), the unet version to train (see models.py)")
    parser.add_argument("--steps", default=28, type=int, help="[int]: (train mode only) Number of steps (batches) per epoch")
    parser.add_argument("--epochs", default=1, type=int, help="[int]: (train mode only) Number of epochs to run training for")
    parser.add_argument("-i", '--image', default=0, type=int, help="[int]: (predict mode only) index of the image to run prediction on")
    parser.add_argument("-u", '--unseen', default=False, type=bool, help="[bool]: (predict mode only) whether to use images found in the unseen (validation) folder")    

    args = parser.parse_args()
    args = vars(args)
    visualise_dir = "./training/"
    just_trained = False

    print("Segmenting roads using both rgb and dtm gradients")

    if (args['mode']  == "train"):
        train(args['steps'], args['epochs'], args['unet_version'])
        just_trained = True
        print("Finished training")


    else:
        if (args['unseen'] == True): #Get our image to predict from the unseen dir
            print("Note- the unseen folder is for illustrative purposes and is not meant as a validation set")
            visualise_dir = "./unseen/"

    visually_inspect_result(just_trained, args['unseen'], args["image"], visualise_dir)
    
    

#Best results: batch size 8, steps per epoch 28 (pass over whole dataset once), epochs 50