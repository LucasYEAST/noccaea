# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:46:19 2020

@author: lucas
"""
import os
import cv2
import numpy as np
from scripts import stats, viz
import random
import pandas as pd


random.seed(69)

PLANT_MULTIMSK_PATH = "data/plant_multimsk/"
PLANT_IMG_PATH = "data/plant_img/"
plant_fns = os.listdir(PLANT_MULTIMSK_PATH)


obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
msk_col_dct = viz.get_colors(obj_class_lst, "Set2")

classpix = {"petiole":[], "margin":[], "vein":[], "tissue":[]}
class_dct = {1:"petiole", 2:"margin", 3:"vein", 4:"tissue"}
class_dct_rev = {v:k for k,v in class_dct.items()}


{'petiole': 1, 'margin': 2, 'vein': 3, 'tissue': 4}




for fn in plant_fns:
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
    assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)

    for msk_class, lst in classpix.items():
        layer_msk = stats.get_layer(multimsk,msk_col_dct,msk_class)
        x,y = np.where(layer_msk > 0 )
        fn_lst = [fn] * len(x)
        class_lst = [class_dct_rev[msk_class]] * len(x)
        lst.extend(list(zip(fn_lst,x,y,class_lst)))

rand1000_classpix = {k : random.sample(v, 1000) for k,v in classpix.items()}
df_lst = []
for lst in rand1000_classpix.values():
    df_lst.extend(lst)
randpix_df = pd.DataFrame(df_lst, columns = ("fn", "x", "y", "pred_class"))
randpix_df.to_csv("data/rand_pred_pixel.csv")

# %% color random pixels
randpix_df = pd.read_csv("data/rand_pred_pixel.csv", index_col=0)
if "obs_class" not in randpix_df.columns:
    randpix_df["obs_class"] = np.nan

randpix_df = randpix_df.sample(frac=1) # Shuffle dataframe

fn_lst = randpix_df.fn.unique().tolist()
for fn in fn_lst:    
    # open image
    img = cv2.imread(PLANT_IMG_PATH + fn) # Load as RGB
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(PLANT_IMG_PATH + fn)
    
    # get random selected pixel coords for this image
    xy = randpix_df.loc[randpix_df.fn == fn,:]
    for _, row in xy.iterrows():
        if not np.isnan(randpix_df.loc[row.name, "obs_class"]):
            continue

        highlight_img = img.copy()
        x,y = row.x, row.y
        highlight_img[x, y] = (255,0,0)
        nb = min(x + 50, highlight_img.shape[0])
        sb = max(0, x - 50)
        eb = max(0, y - 50)
        wb = min(y + 50, highlight_img.shape[1])
        viz.plot_big(highlight_img[sb:nb, eb:wb])
        randpix_df.loc[row.name, "obs_class"] = input("class: ")
        randpix_df.to_csv("data/rand_pred_pixel.csv")

        # viz.plot_big2(img[sb:nb, eb:wb], )

        
    # display image crop & image crop with highlighted pixel
    
    # input class
    
    # save input in dataframe
    
# multimsk_stack = np.stack(multimsks)

