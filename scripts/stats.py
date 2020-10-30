# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:59:53 2020

@author: lucas
"""

import cv2
import numpy as np
import random

def get_layer(plant_multimsk, colors, substructure):
    # TODO change the msk_dct whatever thing so that it makes sense and you don't need 2 dichts
    # pseudo code  
    sub_msk = np.where(plant_multimsk == colors[substructure], (255,255,255), 0)
    return(cv2.cvtColor(sub_msk.astype("uint8"), cv2.COLOR_RGB2GRAY)) 
    
def get_sub_ele_img(plant_ele_img, sub_msk):    
    sub_msk_bool = sub_msk > 0
    return(sub_msk_bool * plant_ele_img)

def get_sub_ele_stats(sub_ele_img):
    abs_ele = sub_ele_img.sum()
    n_pixels = (sub_ele_img > 0).sum()
    mean_C = abs_ele / n_pixels
    return(abs_ele, n_pixels, mean_C)


def insert_class_noise(layer_msk, noise_fraction):
    "changes x % of pixels of one class to a random class"
    N = int(layer_msk.size * noise_fraction)
    X_indices = np.random.randint(0,layer_msk.shape[0], size=N)
    Y_indices = np.random.randint(0, layer_msk.shape[1], size=N)
    layer_msk[X_indices, Y_indices] = 255
    return layer_msk

 

def XrandPixel_value(layer_msk, img, fn, subs, X):
    # Get indices of mask
    x,y = np.where(layer_msk > 0 )
    if len(x) < X:
        print(fn, "substructure: ", subs, "has fewer pixels than X: ", len(x))
        return np.nan
    # Sum random substructure pixel values
    indices = random.sample(range(len(x)), X)
    random_x = x[indices]
    random_y = y[indices]
    return img[random_x,random_y].sum()
    


    


    

