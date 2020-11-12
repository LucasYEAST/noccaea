# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:05:58 2020

@author: lucas
"""
from scripts import utils
import os
import random
from scripts.viz import get_colors
from seaborn import color_palette
import pandas as pd

RAW_TIFF_PATH = "data/raw_data/tiff/"
RAW_TXT_PATH = "data/raw_data/txt/"

BATCH_MSK_PATH = "data/batch_msk/"
BATCH_MULTIMSK_PATH = "data/batch_multimsk/"

PLANT_IMG_PATH = "data/plant_img/"
PLANT_MSK_PATH = "data/plant_msk/"
PLANT_MULTIMSK_PATH = "data/plant_multimsk/"
PLANT_RANDMSK_PATH = "data/plant_randmsk/"

LEAF_MULTIMSK_PATH = "data/leaf_multimsk/"

PLANT_ZIMG_PATH = "data/plant_Zimg/"
PLANT_ZIMG_NOISE_PATH = "data/plant_Zimg_noise/"
PLANT_KIMG_PATH = "data/plant_Kimg/"

DF_SAVE_PATH = "data/Noccaea_CQsA500.csv"
POLY_DCT_PATH = "data/polygon_dict.pck"
LEAFPOLY_DCT_PATH = "data/leaf_polygon_dct.pck"

batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)
plant_fns = os.listdir(PLANT_MSK_PATH)
rand_plant_fns = plant_fns.copy()
random.shuffle(rand_plant_fns)


obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
msk_col_dct = get_colors(obj_class_lst, "Set2")
msk_hex_palette = color_palette(['#%02x%02x%02x' % (msk_col_dct[key][2], msk_col_dct[key][1], msk_col_dct[key][0]) \
                                     for key in obj_class_lst]) # BGR -> RGB -> HEX -> sns palette
msk_hex_palette = dict(zip(obj_class_lst, msk_hex_palette))




leaf_types = ["first", "grown_1", "grown_2", "developping"]
leafmsk_col_dct = get_colors(leaf_types, "hls")


hex_msk_col_dct = {k:'#{:02x}{:02x}{:02x}'.format(v[2],v[1],v[0]) for k,v in msk_col_dct.items()}
RGB_df = pd.Series(hex_msk_col_dct)
RGB_df.to_csv("data/RGB_df.csv")