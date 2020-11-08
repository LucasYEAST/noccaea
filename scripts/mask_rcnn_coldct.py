# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:46:52 2020

@author: lucas
"""
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from main import leaf_types

from seaborn import color_palette

def get_colors(class_lst, color_scheme):
    class_col_dct = {}
    for obj_class, col in zip(class_lst, color_palette(color_scheme)):
        col = [int(x * 255) for x in col]
        col = (col[2], col[1], col[0]) # RGB to BGR for cv2
        class_col_dct[obj_class] = col
    return(class_col_dct)

