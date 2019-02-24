#U-Net for semantically segmenting roads in aerial imagery.
#This version uses rgb data only 

from __future__ import print_function

import os
import cv2
import sys
import math
import util
import time
import keras
import scipy
import psutil
import models
import skimage
import warnings
import argparse
import itertools
import feature_vis
import numpy as np
import random as rn
import skimage.io as io
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
from util import get_single_image, number_of_files, filter_intensity, concatenate_5_layers, load_data, get_model


#Parameters
INPUT_CHANNELS = 3  #RGB
NUMBER_OF_CLASSES = util.NUMBER_OF_CHANNELS #1 Number of mask layers to output. Could be changed to extend this for recognising buildings etc
IMAGE_W = util.IMAGE_W
IMAGE_H = util.IMAGE_H
batch_size = util.BATCH_SIZE #8 Number of images fed to the network at a time
TRAINING_DATA_DIR = util.DATA_SOURCE #Defaults to ./training but could be elsewhere
SAVED_MODEL = "./past_models/rgb_500x14.h5"


"""Loads images from folder without modification, makes them into input batches"""
def batch_generator(batch_size):
    starting_point = 0
    print("Loading files using batch_generator")

    while True:
        image_list = load_data(TRAINING_DATA_DIR+"rgb", batch_size, starting_point) #Get various images out of folders
        mask_list = load_data(TRAINING_DATA_DIR+"mask", batch_size, starting_point)

        with warnings.catch_warnings(): #Resize, rescale mask colours to be black or white
            warnings.simplefilter("ignore")
            images_working = [skimage.transform.resize(image, (IMAGE_H, IMAGE_W)) for image in image_list]
            masks_working = [skimage.transform.resize(mask, (IMAGE_H, IMAGE_W)) for mask in mask_list]
            masks_working = [skimage.color.rgb2gray(i) for i in masks_working]
            masks_working = [skimage.exposure.rescale_intensity(i) for i in masks_working]

        image_list = np.array(images_working, dtype=np.float32)
        mask_list = np.array(masks_working, dtype=np.float32)
        mask_list = mask_list.reshape(batch_size, IMAGE_H*IMAGE_W, NUMBER_OF_CLASSES)
        mask_list = np.nan_to_num(mask_list)

        starting_point += 8 #Move index along to grab next batch
        if(starting_point >= number_of_files()):
            starting_point = 0

        yield image_list, mask_list



"""Uses augmentation to generate additional images to improve the neural net's accuracy"""
def variation_gen(batch_size):
    data_aug = idg( #Transformation parameters- applied to images and masks
        rescale=1./255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True,
        fill_mode="reflect"
    )
    
    img_datagen = data_aug
    mask_datagen = data_aug
    
    shuffle = True #order images are taken from folder- if True, random selection determined by seed
    seed = rn.randint(0,65535) #determines what flips and rotations are applied to each image

    img_gen = img_datagen.flow_from_directory( 
        TRAINING_DATA_DIR,
        classes=["rgb"],
        target_size=(IMAGE_H, IMAGE_W),
        color_mode="rgb",
        class_mode=None,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )

    mask_gen = mask_datagen.flow_from_directory(
        TRAINING_DATA_DIR,
        classes=["mask"],
        target_size=(IMAGE_H, IMAGE_W),
        color_mode="grayscale",
        class_mode=None,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
    print("Loading augmented data")

    while(1):
        imgs= img_gen.next() #Get next batch of transformed images
        masks = mask_gen.next()
        masks = skimage.exposure.rescale_intensity(masks)
        masks = masks.reshape(batch_size,IMAGE_H*IMAGE_W,NUMBER_OF_CLASSES)
        yield imgs, masks




"""Generates predicted mask for an RGB input image"""
def get_prediction(just_trained, index, parent_dir=TRAINING_DATA_DIR):  
    print("Generating predicted mask")
    if (just_trained):
        print("Getting most recent model")
        model = load_model('rgb.h5') #If training, use the most recent model
    else:
        model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL)

    img = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32)[:,:,0:3]

    y_pred = model.predict(img[None,...].astype(np.float32))[0]  #Get prediction from neural net
    y_pred = y_pred.reshape((IMAGE_H,IMAGE_W,NUMBER_OF_CLASSES))
    y_pred = skimage.exposure.rescale_intensity(y_pred)

    return y_pred








"""Shows predicted mask for inspection"""
def visually_inspect_result(just_trained=False, unseen=False, index=0, parent_dir="./training/"):
    print("Getting prediction for visual inspection")
    rgb = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32) #to display

    y_pred = get_prediction(just_trained, index, parent_dir)

    if (unseen == False): #Show ground truth mask and IoU if we are using training data
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
    cv2.imshow('input image',rgb)
    cv2.imshow('RGB result',y_pred)
    
    print("Displaying images")
    cv2.waitKey(0)
    return









"""Trains the unet model using the data generators"""
def train(steps=28, epochs=1, unet=0):
    model = get_model(IMAGE_H, IMAGE_W, INPUT_CHANNELS, SAVED_MODEL, unet)

    callbacks = [
        ProgbarLogger(count_mode='steps', stateful_metrics=None),
        ModelCheckpoint('rgb_weights.h5', monitor='val_loss', save_best_only=False, verbose=0), 
        util.MemLeakCallback()
    ]

    print("Training network")
    history = model.fit_generator(  #Do training from generator
        generator=variation_gen(batch_size),
        steps_per_epoch=steps, #Number of samples obtained using batch_gen or variation_gen (28 for whole dataset)
        validation_data=batch_generator(batch_size),
        epochs=epochs,  #Number of passes over whole dataset
        validation_steps=1,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)

    model.save("./rgb.h5")
    return
        



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='predict', required=True, type=str, help="[train | predict]: Whether to train the network or make predictions on images")
    parser.add_argument("-v", "--unet-version", default=0, type=int, help="[int from 1-3]: (train mode only), the unet version to train (see models.py)")
    parser.add_argument("--steps", default=28, type=int, help="[int]: (train mode only) Number of steps (batches) per epoch")
    parser.add_argument("--epochs", default=1, type=int, help="[int]: (train mode only) Number of epochs to run training for")
    parser.add_argument("-i", '--image', default=0, type=int, help="[int]: index of the image to run prediction on")
    parser.add_argument("-u", '--unseen', default=False, type=bool, help="[bool]: whether to use images found in the unseen (validation) folder")    

    args = parser.parse_args()
    args = vars(args)
    visualise_dir = "./training/" #For making predictions
    just_trained = False  #Whether to use latest model in prediction

    print("Segmenting roads based on RGB data only")

    if (args['mode']  == "train"):
        train(args['steps'], args['epochs'], args["unet_version"])
        just_trained = True
        print("Finished training")

    else:
        if (args['unseen'] == True): #Get our image to predict from the unseen dir
            print("Note- the unseen folder is for illustrative purposes and is not meant as a validation set")
            visualise_dir = "./unseen/"

    visually_inspect_result(just_trained, args['unseen'], args["image"], visualise_dir)

    
#Best results: U-Net 1, batch size 8, steps per epoch 28 (pass over whole dataset once), epochs 500