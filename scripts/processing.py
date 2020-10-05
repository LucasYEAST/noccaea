# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:43:01 2020

@author: lucas
"""
import numpy as np
import cv2

# TODO maybe remove automatic thresholding functions, they seem to be not robust

# Apply masking method using automatic thresholding; https://theailearner.com/tag/image-thresholding/
def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e)//2
    # Left side weight
    w_l = np.sum(b[0][0:i_m+1])
    # Right side weight
    w_r = np.sum(b[0][i_m+1:i_e+1])
    # Until starting point equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) >= i_m:
                w_l += b[0][i_m+1]
                w_r -= b[0][i_m+1]
                i_m += 1
    return i_m

def contouring(img, area_th = 0.0001):
    """Use open-cv contouring to get contours of a binary image.
    
    This function calculates all contours but will return only the contours
    larger than area threshold fraction of the image to avoid noise.
    Input: img, a numpy array containing a binary image.
    Returns: large_contours, open-cv contours larger than 1 percent of the image.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    x , y = np.shape(img)
    total_area = x * y
    for c in contours:
        if cv2.contourArea(c) > int(total_area * area_th):
            large_contours.append(c)
    large_contours = np.array(large_contours)
    return large_contours

def create_mask(img, contours):
    """This function will create a mask based on the contours and the original image.
    
    Input: img, a numpy array with an image.
    Input: contours, open-cv contours.
    Returns: binary_mask, the contours drawn on a black image. Not actually binary since black is 0 and white is 255.
    """
    binary_mask = np.zeros(np.shape(img), dtype=np.uint8)
    cv2.drawContours(binary_mask, contours, -1, (255,255,255), -1)
    return binary_mask

# Python program to extract rectangular 
# Shape using OpenCV in Python3 https://www.geeksforgeeks.org/python-draw-rectangular-shape-and-extract-objects-using-opencv/

# mouse callback function 
def draw_circle(event, x, y, flags, param): 
	global ix, iy, drawing, mode 
	
	if event == cv2.EVENT_LBUTTONDOWN: 
		drawing = True
		ix, iy = x, y 
	
	elif event == cv2.EVENT_MOUSEMOVE: 
		if drawing == True: 
			if mode == True: 
				cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3) 
				a = x 
				b = y 
				if a != x | b != y: 
					cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1) 
			else: 
				cv2.circle(img, (x, y), 5, (0, 0, 255), -1) 
	
	elif event == cv2.EVENT_LBUTTONUP: 
		drawing = False
		if mode == True: 
			cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2) 
	
		else: 
			cv2.circle(img, (x, y), 5, (0, 0, 255), -1) 
	


