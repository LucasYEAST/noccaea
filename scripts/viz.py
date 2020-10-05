# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:41:26 2020

@author: lucas
"""
import matplotlib.pyplot as plt
from seaborn import color_palette
import cv2


# colors = {}
# for k,v in mcolors.BASE_COLORS.items(): # TODO: improve, this is ugly
#     col_lst = list(v)
#     for i,col in enumerate(col_lst):
#         col_lst[i] = int(col * 255)
#     colors[k] = tuple(col_lst)
# col_keys = list(colors.keys())

# msk_colors = {msk_name : col for msk_name, col in zip(sub_structures, col_keys)}

# Overlay viz parameters
alpha = 0.7
beta = (1.0 - alpha)

def get_colors(class_lst):
    class_col_dct = {}
    for obj_class, col in zip(class_lst, color_palette("colorblind")):
        col = [int(x * 255) for x in col]
        col = (col[2], col[1], col[0]) # RGB to BGR for cv2
        class_col_dct[obj_class] = col
    return(class_col_dct)
        

def plot_big(img, title=""):
    """plot 10x10 image in grayscale or color"""
    plt.figure(figsize=(10,10))
    plt.title(title)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()

# def plot_big(img, title=""):
# 	"""Plot 10x10 image in color or grayscale"""
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(10,10))
# 	plt.imshow(img, cmap='gray')
# 	plt.title(title)
# 	plt.show()
    
def plot_big2(img0, img1, title0="", title1=""):
    """plot 2 10x10 images next to each other"""
    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(img0, cmap='gray')
    plt.title(title0)
    plt.subplot(2,2,2)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.show()	
    
def overlay(img1, img2, alpha = alpha, beta = beta, title=""):
    # Creates overlay and plots image 
    if (img1.dtype != "uint8"):
        print("converting img1 to uint8")
        img1 = img1.astype("uint8")
    
    if (img2.dtype != "uint8"):
        print("converting img2 to uint8")
        img2 = img2.astype("uint8")
    
    overlay_img = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    plot_big(overlay_img, title=title)
    return overlay_img


