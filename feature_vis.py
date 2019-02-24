"""
Feature/algorithm visualisation code

Finds input to maximise a particular layer of the trained network- either the output or an intermediate filter.
Essentially runs network in reverse to find what its idea of a road looks like.

Hacked together from code at https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
"""

import keras
from keras import backend as K
import numpy as np
import skimage
import cv2
from keras.models import load_model
import scipy
import util
from util import concatenate_5_layers, get_5layer_img
import random 
from array2gif import write_gif



model2 = "clnet_1000x28.h5" #So we can put the model name in the name of the images it generates
model1="../roads_backups/old_models/"+model2


def filter_vis_experiment(num_images, iters, test_model=model1): #Uses gradient ascent to find the input that maximises a particular part of the convnet

    print("Convolution filter visualisation experiment")
    
    model = load_model(test_model)
    input_img = model.layers[0].input

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = 'conv2d_17'
    filter_index = 0

    for j in range(num_images):
        #Maximise intermediate Conv filters
        """
        filter_index = j
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, j])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        
        #Maximise output layer 
        """
        loss = K.mean(model.output[:,:,:])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        
        
        iterate = K.function([input_img], [loss, grads])
        # run gradient ascent for 20 steps
        #input_img_data = np.zeros((1, 256, 256, 5))
        #input_img_data = np.random.random((1, 256, 256, 5)) * 1.0 #random noise input
        input_img_data = np.expand_dims(util.get_5layer_img(j*j), 0)   #existing image input


        for i in range(iters):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data = np.squeeze(input_img_data)
            input_img_data = cv2.GaussianBlur(input_img_data, (3,3), 5/(i+1)) 
            input_img_data = np.expand_dims(input_img_data, 0)
            input_img_data += grads_value * 1.0
            print("%d\r" % i)

            if(i != 0 and i%100 == 0):

                img = input_img_data[0] 
                img = deprocess_image(img)
                """
                dx = img[:, 3, :]
                dy = img[:,4,:]
                img = img[:,0:3,:]

                dx = np.squeeze(dx) 
                dy = np.squeeze(dy)
                """
                print("                  ", j)

                img = np.swapaxes(img, 1, 2)

                #img[:,:,0:3] = cv2.cvtColor(img[:,:,0:3], cv2.COLOR_RGB2BGR)
                y_pred = prediction(img)

                cv2.imshow("pred", y_pred)
                cv2.imshow("rgb", img[:,:,0:3])

                cv2.waitKey(0)
                scipy.misc.imsave("./results/"+model2+"_"+str(i)+"_"+str(j)+"max.png", img[:,:,0:3])


    return


def prediction(img):
    model = load_model(model1)
    y_pred = model.predict(img[None,...].astype(np.float32))[0]    
    y_pred = y_pred.reshape((256,256,1))
    return y_pred

def five_layer_bilateral(img): #Runs bilateral noise filtering on a 5 layer image, RGB/dx/dy done separately
    sig_col = 0.5
    sig_space = 0.01

    img[img < 0] = 0

    dx = img[:, :,3]
    dy = img[:,:,4]
    img = img[:,:,0:3]
    
    dx = np.squeeze(dx) 
    dy = np.squeeze(dy) 

    img = skimage.restoration.denoise_bilateral(img, sigma_color=sig_col, sigma_spatial=sig_space)
    dx = skimage.restoration.denoise_bilateral(dx, multichannel=False, sigma_color=sig_col, sigma_spatial=sig_space)
    dx = np.expand_dims(dx, axis=2)
    dy = skimage.restoration.denoise_bilateral(dy, multichannel=False, sigma_color=sig_col, sigma_spatial=sig_space)
    dy =np.expand_dims(dy, axis=2)

    output = concatenate_5_layers(img, dx, dy,2)
    return output




def deprocess_image(x): #Manipulates image to make it nicer to look at
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x



if __name__ == '__main__': 
    filter_vis_experiment(1, 501)
