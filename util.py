"""File for code that is shared/redundant between classifier scripts- includes functions for loading/manipulating
    data and other useful functions for evaluating performance."""

import keras
import os
import skimage
import numpy as np
import sys
import scipy
import warnings
import cv2
import matplotlib.pyplot as plt
import psutil
import random as rn
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2D, Reshape
import models
from keras.layers.core import Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop, SGD
from keras.models import Model
from keras.models import load_model

"""Overall parameters"""
IMAGE_H = 256
IMAGE_W = 256
NUMBER_OF_CHANNELS = 1 #Number of channels in output. Could be more than 1 if we want to segment different classes in addition to roads.
BATCH_SIZE = 8
DATA_SOURCE = "./training/"


"""General code"""

def get_model(height, width, channels, model_path, version=0): #Gets a new, untrained unet, or a pretrained model. Returns Keras model object
    inputs = Input((height, width, channels))

    if (version == 1): #Untrained u-net implementations (see models.py)
        base = models.get_unet(inputs, 1)
    elif (version == 2):
        base = models.get_unet2(inputs, 1)
    elif (version == 3):
        base = models.get_unet3(inputs, 1)
    else:
        print("Getting pretrained model from %s" % model_path)
        model = load_model(model_path)
        return model

    reshape= Reshape((-1,1))(base)
    act = Activation('relu')(reshape)
    
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adadelta(), loss="binary_crossentropy")            
 
    return model



"""Utility functions for 5 layer images, getting stuff from folders, etc"""

def load_data(data_directory, batch_size, starting_point, verbose=False): #Get whole batch of 5 channel images at once
    images = []
    file_names = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".png")]
    file_names = sorted(file_names) #Get list of all files then index that
    for f in file_names[starting_point:]:
        images.append(skimage.data.imread(f))
        #print(len(images), batch_size)
        if(verbose):
            print("got image %s" % f)
        if (len(images) == batch_size):
            return images
        

def get_single_image(parent_dir, directory, index): #Get png from folder with specified index
    load_dir = parent_dir+directory
    #print(load_dir)
    file_names = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if f.endswith(".png")]
    file_names = sorted(file_names)
    if (file_names == None or index > len(file_names)):
        print("Couldn't find image number", index, " in directory ", load_dir)
        sys.exit(1)
    #print(file_names[index])
    img = skimage.data.imread(file_names[index])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.transform.resize(img, (IMAGE_H, IMAGE_W))

    return img


def get_5layer_img_old(index, parent_dir=DATA_SOURCE):  #Obsolete function for getting dxy gradients from png files
    print("This is obsolete because it gets a 5 layer image using png files. If you're using 5layer_classifier.py, use get_5layer_img instead")
    img = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32)
    dx = np.array(get_single_image(parent_dir, "x_grad", index), dtype=np.float32)
    dy = np.array(get_single_image(parent_dir, "y_grad", index), dtype=np.float32)
    img = concatenate_5_layers(img, dx, dy, 2)
    return img

def get_5layer_img(index, parent_dir=DATA_SOURCE):  #Gets dxy from raw .txt heightmap data- more consistent and accurate than get_5layer_img_old
    img = np.array(get_single_image(parent_dir, "rgb", index), dtype=np.float32)
    dxy = get_single_dxy(parent_dir, index)
    #print(img.shape, dxy.shape)
    img = np.concatenate((img, dxy), 2)
    return img


def number_of_files(dir=DATA_SOURCE+"rgb"): #Number of training images
    return len([name for name in os.listdir(dir)])

def concatenate_5_layers(img, dx, dy, axis): #[Obsolete] Helper function to save rewriting code
    if (dy.ndim != img.ndim):
        dx = np.expand_dims(dx, axis=3) #adds another dimension if needed
        dy = np.expand_dims(dy, axis=3) 

    img = np.concatenate((img, dy), axis=axis)
    img = np.concatenate((img, dx), axis=axis)
    return img

def filter_intensity(pred, threshold): #Makes everything below a threshold zero
    pred2 = np.copy(pred)
    #pred2[pred > threshold] = 1
    pred2[pred <= threshold] = 0

    return pred2
   
def to_rgb1(im): #Convert grayscale to rgb in a simple logical way
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def plot_loss_graph(history): #Plots a line graph of the loss at each epoch (takes Keras history object)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return


def compare_pred_mask(y_pred, mask): #Sums all pixel intensities for prediction and ground truth and shows how close they are (as a %)
    y_pred = np.squeeze(y_pred)

    pred_sum = float(np.sum(y_pred))
    mask_sum = float(np.sum(mask))
    if(np.isnan(mask_sum)):#means something is wrong
        mask_sum = 1.0

    diff = np.abs(mask-y_pred) 
    wrong = float(np.sum(diff))

    similarity = (pred_sum*100/mask_sum)
    abs_diff = (wrong*100/mask_sum)

    if(np.isnan(abs_diff)): #means mask_sum = 0 ie. no roads visible
        abs_diff = 0.0

    #print("Raw result sum = ", pred_sum, "Ground truth sum =", mask_sum, "absolute diff", wrong)
    print("Naive similarity percent = %f" % similarity) #sum(predicted) as a % of sum(truth)
    print("Abs. diff. percent = %f" % abs_diff) #sum(abs.diff.) as a % of sum(truth)
    #cv2.imshow("difference", diff)

    return similarity, abs_diff


def show_layers(img): #Shows what's on each layer of a 5 layer image
    cv2.imshow("rgb channels", np.flip(img[:,:,0:3], 2))
    cv2.imshow("dx channel", img[:,:,3])
    cv2.imshow("dy channel", img[:,:,4])
    cv2.waitKey(0)
    return


def intersection_over_union(y_pred, mask, write_to_file=False): #Metric for measuring accuracy of a semantic segmentation attempt
    threshold = 0.3 #Good balance between signal and noise
    y_pred[y_pred < threshold] = 0
    y_pred[y_pred > threshold] = 1
    y_pred = np.squeeze(y_pred)
    mask = np.nan_to_num(mask)

    intersection = np.array(np.logical_and(y_pred, mask), dtype=np.uint8)
    union = np.array(np.logical_or(y_pred, mask), dtype = np.uint8)
    #print(np.sum(intersection), np.sum(union))

    total = np.sum(intersection)/float(np.sum(union))
    if(np.sum(union) == 0.0): #Because otherwise the case where both are 0 gives iou=0
        total = 1.0
    if(write_to_file):
        f = open("./ious", "a")
        f.write("%f\n" % total)
        f.close()

    return total


"""Callbacks"""


"""Callback to record how much memory is really being used by the program (psutil "unique set size")"""
class MemLeakCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.process = psutil.Process(os.getpid())

    def on_epoch_end(self, epoch, logs={}):
        percentage = self.process.memory_percent(memtype='uss') #uss is supposed to be most accurate real world reflection of memory usage
        print("Using %f percent of memory" % percentage)







"""Code for getting x/y gradients from dtm txt files"""

def batch_from_txt(label_directory, batch_size, starting_index, verbose=False): #Return a batch of 2 channel dxy tiles
    dx_list = []
    dy_list = []
    file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".txt")]
    file_names = sorted(file_names)
    for i in range(batch_size):
        dx, dy = get_dxy_from_txt(file_names[starting_index+i], verbose)
        dx_list.append(dx)
        dy_list.append(dy)

    dx_list = np.array(dx_list) #convert list to ndarray and make 4d
    dy_list = np.array(dy_list)

    #print(dx_list.shape, "               ", dy_list.shape)
    output = np.stack((dx_list, dy_list), axis=-1)

    return output


def get_single_dxy(parent_dir, index): #Wrapper to make it behave in the same way as getsingleimg- more versatile
    file_names = [os.path.join(parent_dir+"txt", f) for f in os.listdir(parent_dir+"txt") if f.endswith(".txt")]
    file_names = sorted(file_names)
    txtfile = file_names[index]
    dx, dy = get_dxy_from_txt(txtfile)
    output = np.stack((dx, dy), axis=-1)
    return output



def get_dxy_from_txt(file, verbose=False): #Generates dx and dy tiles from a single text file
    data = np.genfromtxt(file, delimiter=' ', usecols=(2))
    data = np.reshape(data, (256, 256), order="C")
    data = np.flipud(data) 
    grady = np.gradient(data, axis=0)
    gradx = np.gradient(data, axis=1)

    stddx = np.std(gradx)
    stddy = np.std(grady)
    medx = np.median(gradx)
    medy = np.median(grady)
    gradx[gradx > stddx*10] = medx #Points with a huge gradient (usually water artifacts) are averaged out
    grady[grady > stddy*10] = medy #Huge gradient = 10 standard deviations above average (shouldn't happen naturally)
    gradx[gradx < -stddx*10] = medx
    grady[grady < -stddy*10] = medy

    #np.set_printoptions(threshold=np.inf)
    #print(gradx, grady)
    #print(np.max(gradx), np.max(grady), np.min(gradx), np.min(grady), stddx , stddy )
    if(verbose):
        print("Loaded dx and dy gradients from file %s" % file)
        scipy.misc.imshow(gradx)
        scipy.misc.imshow(grady)
    return gradx, grady



if __name__ == '__main__': #To check that layers match up
    index = sys.argv[1]
    print(index)
    if(index == None):
        index = 0
    show_layers(get_5layer_img(int(index)))
