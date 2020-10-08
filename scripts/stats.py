# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:59:53 2020

@author: lucas
"""

import cv2
import numpy as np

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

def calc_CQ(mean_C_plant, mean_C_sub):
    return mean_C_sub/mean_C_plant
    


    

