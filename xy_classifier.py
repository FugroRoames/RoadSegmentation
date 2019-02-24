#U-Net for semantically segmenting roads in aerial imagery.
#This version uses DTM gradients only 

import os
import cv2
import sys
import math
import time
import util
import keras
import scipy
import models
import psutil
import skimage
import warnings
import argparse
import feature_vis
import numpy as np
import random as rn
import skimage.io as io
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import concatenate
import skimage.morphology as morphology
from util import batch_from_txt, get_dxy_from_txt
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
from util import get_single_image, number_of_files, filter_intensity, concatenate_5_layers, load_data, get_model


#Parameters
INPUT_CHANNELS = 2  #dx/dy
NUMBER_OF_CLASSES = util.NUMBER_OF_CHANNELS #1 Number of mask layers to output. Could be changed to extend this for recognising buildings etc
IMAGE_W = util.IMAGE_W
IMAGE_H = util.IMAGE_H
batch_size = util.BATCH_SIZE #8 Number of images fed to the network at a time
TRAINING_DATA_DIR = util.DATA_SOURCE #Defaults to ./training but could be elsewhere
SAVED_MODEL = "./past_models/dxy_500x28.h5"
   

"""Loads images from computer, makes them into input batches"""
def batch_generator(batch_size):
    starting_point = 0
    print("Loading text files using batch_generator")   

    while True:
        image_list = batch_from_txt(TRAINING_DATA_DIR+"txt", batch_size, starting_point) #Get 256x256 tiles from txt files
        mask_list = load_data(TRAINING_DATA_DIR+"mask", batch_size, starting_point)

        with warnings.catch_warnings(): #Reshape and rescale masks to be black or white
            warnings.simplefilter("ignore")
            mask_list = [skimage.transform.resize(mask, (IMAGE_H, IMAGE_W)) for mask in mask_list]
            mask_list = [skimage.color.rgb2gray(i) for i in mask_list]
            mask_list = [skimage.exposure.rescale_intensity(i) for i in mask_list]

        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list = mask_list.reshape(batch_size, IMAGE_H*IMAGE_W, NUMBER_OF_CLASSES)
            
        starting_point += batch_size #Move index along to grab next batch
        if(starting_point >= number_of_files()):
            starting_point = 0

        yield image_list, mask_list


"""Generates augmented image/mask pairs from txt data"""
def from_txt_generator(batch_size): 
    print("Getting augmented data from text files")
    index = 0
    img_datagen = idg( #Transformation params
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True,
        fill_mode="reflect"
        )
    
    while(1):
        seed = rn.randint(0,5000)
        output = batch_from_txt(TRAINING_DATA_DIR+"txt", batch_size, index, False) #Get batch of dxy images from txt
        mask_list = load_data(TRAINING_DATA_DIR+"mask", batch_size, index) #Get masks

        with warnings.catch_warnings(): #Rescale etc
            warnings.simplefilter("ignore")
            mask_list = [skimage.transform.resize(mask, (IMAGE_H, IMAGE_W)) for mask in mask_list]
            mask_list = [skimage.color.rgb2gray(i) for i in mask_list]
            mask_list = [skimage.exposure.rescale_intensity(i) for i in mask_list]

        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list = np.expand_dims(mask_list, 3)
        mask_list = np.nan_to_num(mask_list) #Need this for some reason???

        index += batch_size #Move index along to grab next batch
        if(index >= number_of_files()):
            index = 0
            
        dx_gen = img_datagen.flow(output, mask_list, batch_size=batch_size, shuffle=True, seed=seed)
        mask_gen = img_datagen.flow(mask_list, mask_list, batch_size=batch_size, shuffle=True, seed=seed)
    
        dxy = dx_gen.next()[0] #Obtain a batch of transformed images 
        mask = mask_gen.next()[0] #The flow method produces a tuple of images + labels, but we don't want labels for this task
        
        """cv2.imshow("dx1", dxy[0,:,:,0])
        cv2.imshow("dx2", dxy[1,:,:,0])
        cv2.imshow("m1", np.reshape(mask[0], (256,256)))
        cv2.imshow("m2", np.reshape(mask[1],(256,256)))
        cv2.waitKey(0)"""

        mask = mask.reshape(batch_size, IMAGE_H*IMAGE_W, NUMBER_OF_CLASSES)

        yield dxy, mask











"""Generate predicted mask for a 2 channel dxy input tile"""
def get_prediction(just_trained, index, parent_dir=TRAINING_DATA_DIR):
    print("Generating predicted mask")
    if (just_trained):
        print("Getting most recent model")
        model = load_model('dxy.h5') #If training, use most recent model (autosaved as dxy.h5)
    else:
        model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL)

    
    file_names = [os.path.join(parent_dir+"txt", f) for f in os.listdir(parent_dir+"txt") if f.endswith(".txt")]
    file_names = sorted(file_names)
    dx, dy = get_dxy_from_txt(file_names[index])
    img = np.stack((dx, dy), axis=-1)

    y_pred = model.predict(img[None,...].astype(np.float32))[0]  #Get prediction
    y_pred = y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
    y_pred = skimage.exposure.rescale_intensity(y_pred)

    return y_pred





"""Visually inspect predicted mask"""
def visually_inspect_result(just_trained, unseen=False, index=0, parent_dir=TRAINING_DATA_DIR):
    print("Commencing visual inspection of result")
    rgb = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32)
    
    file_names = [os.path.join(parent_dir+"txt", f) for f in os.listdir(parent_dir+"txt") if f.endswith(".txt")]
    file_names = sorted(file_names)
    dx, dy = get_dxy_from_txt(file_names[index])

    y_pred = get_prediction(just_trained, index, parent_dir)

    if(unseen == False): #Get mask and IoU if using training data
        mask = get_single_image(parent_dir, "mask", index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = skimage.color.rgb2gray(mask)
            mask = skimage.exposure.rescale_intensity(mask)
            cv2.imshow('ground truth',mask)

        y_pred2 = np.copy(y_pred)
        iou = util.intersection_over_union(y_pred2, mask)
        print("IoU = %f" % iou)

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    #cv2.imshow("rgb", rgb)
    cv2.imshow("input dx",dx)
    cv2.imshow('X/Y result',y_pred)
 
    print("Displaying images")
    cv2.waitKey(0)
    return




"""Trains the unet model using the data generators"""
def train(steps=28, epochs=1, unet=0):
    model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL, unet)

    callbacks = [
        ProgbarLogger(count_mode='steps', stateful_metrics=None),
        ModelCheckpoint('dxy_weights.h5', monitor='val_loss', save_best_only=False, verbose=0),
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

    model.save("./dxy.h5")
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

    print("Segmenting roads using pixel-wise dtm gradients only")

    
    if (args["mode"] == 'train'):
        train(args['steps'], args['epochs'], args['unet_version'])
        just_trained = True
        print("Finished training")

    else:
        if (args['unseen'] == True): #Get our image to predict from the unseen dir
            print("Note- the unseen folder is for illustrative purposes and is not meant as a validation set")
            visualise_dir = "./unseen/"

    visually_inspect_result(just_trained, args['unseen'], args["image"], visualise_dir)
    

#Best results: batch size 8, steps per epoch 28 (pass over whole dataset once), epochs 500
