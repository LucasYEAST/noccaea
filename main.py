# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:40:58 2020

@author: lucas
"""

import pickle
import os

import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from scripts import utils, segmentation, stats, viz

sns.set(font_scale=1.3)
sns.set_style("ticks")

np.random.seed(69)

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

POLY_DCT_PATH = "data/polygon_dict.pck"
LEAFPOLY_DCT_PATH = "data/leaf_polygon_dct.pck"

# Download data
assert os.path.exists(RAW_TIFF_PATH), "Request data from lucas.vanderzee@wur.nl"

# Get list of all filenames of scans (batches)
batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)

# Get list of all filenames of plants
try:
    plant_fns = os.listdir(PLANT_MSK_PATH)
except OSError:
    print("plant files not created yet")


# Define classes and their associated colors
obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
class_dct = {1:"petiole", 2:"margin", 3:"vein", 4:"tissue"}
class_dct_rev = {v:k for k,v in class_dct.items()}
msk_col_dct = viz.get_colors(obj_class_lst, "Set2")

# Define leaf age classes and color
leaf_types = ["first", "grown_1", "grown_2", "developping"]
leafmsk_col_dct = viz.get_colors(leaf_types, "hls")

#TODO Write code to download data

# %% Manually divide batches into plants and annotate
# df = pd.read_csv("data/Noccaea_nometrics.csv", index_col=0)

# if os.path.exists(POLY_DCT_PATH):
#     input("Are you sure you want to override/extend the existing plant polygons? (any key / Ctrl-C to abort)")
#     with open(POLY_DCT_PATH, "rb") as f:
#         polygon_dct = pickle.load(f)
# else:
#     polygon_dct = {batchname:{} for batchname in batchname_lst}

# # draw plant polygons
# for batch in batchname_lst:
#     df = segmentation.divide_plants(RAW_TIFF_PATH, batch, polygon_dct, POLY_DCT_PATH, 
#                                     df, "data/Noccaea_nometrics.csv")

# %% Create batch foreground masks
if not os.path.exists(BATCH_MSK_PATH):
    os.mkdir(BATCH_MSK_PATH)

layers = ["Ca.tif", "K.tif", "Ni.tif",] # "Image.tif"
for batch in batchname_lst:
    segmentation.create_foreground_masks(RAW_TIFF_PATH, batch, layers, BATCH_MSK_PATH)
       
# %% Create batch segmentation masks

blade_ksize, lap_ksize, thin_th, fat_th = 15, 7, -2500, 0
if not os.path.exists(BATCH_MULTIMSK_PATH):
    os.mkdir(BATCH_MULTIMSK_PATH)

for batch in batchname_lst:
    multimsk = segmentation.create_multimsks(batch, RAW_TIFF_PATH, BATCH_MSK_PATH, 
                     blade_ksize, lap_ksize, thin_th, fat_th, msk_col_dct, 
                     path = BATCH_MULTIMSK_PATH)
    
# %% Create individual images per plant
metals = ["Zn"]
with open(POLY_DCT_PATH, "rb") as f:
    polygon_dct = pickle.load(f)

# Create output dictionaries
paths = [PLANT_IMG_PATH, PLANT_MSK_PATH, PLANT_MULTIMSK_PATH] + \
    ["data/plant_" + metal + "img/" for metal in metals]
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)

for batch in batchname_lst:
    
    # Open batch masks and images
    msk_path = BATCH_MSK_PATH + batch + "batchmsk.tif"
    msk = cv2.imread(msk_path,  cv2.IMREAD_GRAYSCALE) 
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(msk_path)
    
    img_path = RAW_TIFF_PATH + batch + "- Image.tif"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(img_path)
    
    multimsk_path = BATCH_MULTIMSK_PATH + batch + "multimsk.tif"
    multimsk = cv2.imread(multimsk_path) # Load as RGB
    assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(multimsk_path)
            
    # Crop individual plants from batch images
    for acc_rep, polygon in polygon_dct[batch].items():
        # Create mask/image name
        accession, replicate = acc_rep.split("_")
        fn = "_".join([batch, accession, replicate])
        
        plant_msk = segmentation.crop_plant(polygon, msk, msk, bg_color = None)
        cv2.imwrite(PLANT_MSK_PATH + fn + ".tif", plant_msk)
        
        plant_img = segmentation.crop_plant(polygon, img, msk, bg_color = None)
        cv2.imwrite(PLANT_IMG_PATH + fn + ".tif", plant_img)
        
        plant_multimsk = segmentation.crop_plant(polygon, multimsk, msk, bg_color = msk_col_dct['background'])
        cv2.imwrite(PLANT_MULTIMSK_PATH + fn + ".tif", plant_multimsk)

        
    # Crop individual plants from raw metal concentration files
    for metal in metals:
        for acc_rep, polygon in polygon_dct[batch].items():
            accession, replicate = acc_rep.split("_")
            fn = "_".join([batch, accession, replicate])
            
            metalimg_path = RAW_TXT_PATH + batch + "- " + metal + ".txt"    
            batch_metalimg = np.loadtxt(metalimg_path, delimiter=",", skiprows=1)
            plant_metal_img = segmentation.crop_plant(polygon, batch_metalimg, msk, bg_color = None)
            np.savetxt("data/plant_" + metal + "img/" + fn + ".txt", plant_metal_img, fmt='%f', delimiter=",")

# %% Create masks with x percentage of pixels replaced by a random class
plant_fns = os.listdir(PLANT_MSK_PATH)
for percentage in [10,20,50,75,90,100]: 
    segmentation.create_noised_msks(PLANT_MULTIMSK_PATH, PLANT_MSK_PATH, plant_fns, msk_col_dct, percentage)


# %% Create random substructures with (N_pixels) amount of random substructures per plant
for N_pixels in [1,2,5]:
    segmentation.create_rand_substructure(PLANT_MSK_PATH, PLANT_RANDMSK_PATH, N_pixels)

# %% Get stats from image
# TODO: profile time usage and improve speed

metals = ["metal_Zn", "metal_K", "metal_Ni", "metal_Ca"]
substructures = obj_class_lst[1:] + ["plant", "rand_1", "rand_2", "rand_5"]
df = pd.read_csv("data/Noccaea_nometrics.csv", index_col=0)

plant_fns = os.listdir(PLANT_MSK_PATH)
# TODO noise levels loop 
# noise loop
# for nlvl in ["10", "20", "50", "75", "90"]:

# plant loop
for fn in plant_fns:
    # load plant mask and multi-mask
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
    
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
    assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
       
    # metals loop
    for metal in metals:
        # load metal image
        metal_name = metal.split("_")[1]
        METAL_PATH = "data/plant_" + metal_name + "img/"
        img = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
    
        # substructure loop
        for substrct in substructures:
           # load correct mask
           if substrct == "plant":
               layer_msk = msk
           elif substrct == "rand_1":
               layer_msk = cv2.imread(PLANT_RANDMSK_PATH + "1/" + fn, cv2.IMREAD_GRAYSCALE)
           elif substrct == "rand_2":
               layer_msk = cv2.imread(PLANT_RANDMSK_PATH + "2/" + fn, cv2.IMREAD_GRAYSCALE)
           elif substrct == "rand_5":
               layer_msk = cv2.imread(PLANT_RANDMSK_PATH + "5/" + fn, cv2.IMREAD_GRAYSCALE)
           elif substrct in obj_class_lst[1:]:
               layer_msk = stats.get_layer(multimsk, msk_col_dct, substrct)
           else:
               raise Exception("substructure: " + substrct + " is invalid")
           subs_metal_image = stats.get_sub_ele_img(img, layer_msk)
           abs_metal, n_pixels, meanC = stats.get_sub_ele_stats(subs_metal_image)
           # A500 = stats.XrandPixel_value(layer_msk, img, fn, substrct, 500)
           colnames = ["_".join((metal, substrct, metric)) for metric in ["abs", "n_pix", "meanC", ]] #"A500"
           df.loc[df["fn"] == fn, colnames] = [abs_metal, n_pixels, meanC, ] #A500

        # Calculate CQ for all substructures
        CQ_colnames = ["_".join((metal, substrct, "CQ")) for substrct in substructures]
        mean_colnames = ["_".join((metal, substrct, "meanC")) for substrct in substructures]
        plant_mean_colname = "_".join((metal, "plant", "meanC"))
        df[CQ_colnames] = df[mean_colnames].div(df[plant_mean_colname], axis=0)
df.to_csv("data/Noccaea_CQs.csv")

# %% Sensitivity analysis prep

# Set parameter values to test
para_dct={"blade_ksize": [7, 11, 19, 23],
		"lap_ksize": [ 3, 5, 7, 9, 11],
		"thin_th":  [-3500, -3000 ,-2500, -2000, -1500],
		"fat_th":	[-200, -100, 0, 100, 200]}

# Set non-varying parameter values
b= 15
l= 7
t= -2500
f= 0

# Create dataframe with parameter values
para_df = pd.DataFrame(columns=para_dct.keys())
para_map = {}
i = 0
for k,v in para_dct.items():
    for para in v:
        para_df.loc[i,:] = [b,l,t,f]
        para_df.loc[i, k] = para
        para_map[i] = k + "_" + str(para)
        i += 1
para_df.to_csv("data/sensitivity_paras.csv")

# %% Sensitivity analysis

# Load ground truth and parameter values to test
ground_truth_path = "data/rand_pred_pixel.csv"
assert os.path.exists(ground_truth_path), "ground truth not found, make sure curated_pixels.py was run"
randpix_df = pd.read_csv(ground_truth_path, index_col=0, header=0)
para_df = pd.read_csv("data/sensitivity_paras.csv", index_col=0, header=0)

# Load plant polygons to separate batch multimasks
with open(POLY_DCT_PATH, "rb") as f:
    polygon_dct = pickle.load(f)

# Create column with (x,y) values for ground truth pixels
xy_tuplst = list(zip(randpix_df.x.tolist(), randpix_df.y.tolist()))
randpix_df["xy"] = xy_tuplst

# Loop over parameter variations and document predicted classes for ground-truth pixels
for i in range(len(para_df)):
    blade_ksize, lap_ksize, thin_th, fat_th = para_df.loc[i, ['blade_ksize', 'lap_ksize', 'thin_th', 'fat_th']]
    randpix_df["pred_class_" +str(i)] = np.nan

    for batch in batchname_lst:
        # Create segmentation mask using the parameter values
        multi_msk = segmentation.create_multimsks(batch, RAW_TIFF_PATH, BATCH_MSK_PATH, 
                             blade_ksize, lap_ksize, thin_th, fat_th,
                             msk_col_dct, BATCH_MULTIMSK_PATH)
        
        # Divide batch multimask into individual plants
        for acc_rep, polygon in polygon_dct[batch].items():
            accession, replicate = acc_rep.split("_")
            fn = "_".join([batch, accession, replicate])
            fn += ".tif"
            
            # Remove other plants
            bged_multimsk = segmentation.poly_crop(multi_msk, polygon, 
                                             col = (255,255,255), bg = msk_col_dct['background'])
            
            # Crop image to bounding box around polygon
            x,y,w,h = cv2.boundingRect(polygon)
            plant_multimsk = bged_multimsk[y:y+h,x:x+w]
            
            # Get pixel class from adusted multimask for the 4000 surveyed pixels
            for substrct in obj_class_lst[1:]:
                layer_msk = stats.get_layer(plant_multimsk,msk_col_dct,substrct)
                x,y = np.where(layer_msk > 0 )
                xy_lst = list(zip(x,y))
                randpix_df.loc[(randpix_df.fn == fn) & (randpix_df.xy.isin(xy_lst)),"pred_class_"+str(i)] = class_dct_rev[substrct]             
randpix_df.to_csv("data/rand_pred_pixel_sens.csv")
# %%
### Below all code is written for leaf age segmentation. This part is not described in the article
# %% Manually annotate leaf age

# Load dictionary of already segmented leaves
# if path.exists(LEAFPOLY_DCT_PATH):
#     input("are you sure you want to extend/overwrite current leaf polygon dictionary? (any key/Ctrl-C to abort")
#     with open(LEAFPOLY_DCT_PATH, "rb") as f:
#         leaf_polygon_dct = pickle.load(f)
# else:
#     leaf_polygon_dct = {}

# # Randomize order of annotation
# rand_plant_fns = plant_fns.copy()
# random.shuffle(rand_plant_fns)

# # Manually annotate leaf age
# for fn in rand_plant_fns:
#     print("working on: ", fn)
    
#     # Set up dictionary or skip if plant already annotated
#     if fn not in leaf_polygon_dct:
#         leaf_polygon_dct[fn] = {leaftype:[] for leaftype in leaf_types}
#         leaf_polygon_dct[fn]["status"] = "pending"
#     elif leaf_polygon_dct[fn]["status"] == "pending":
#         for k,v in leaf_polygon_dct[fn].items():
#             print(k, ": ", len(v))
#     elif leaf_polygon_dct[fn]["status"] == "done":
#         print("already annotated")
#         continue
#     else:
#         raise Exception('Non accepted status encountered')
        
#     # Load plant image and mask        
#     img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
#     assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    
#     multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
#     assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
    
#     # Create blade mask from margin, vein and tissue masks
#     blade_substructs = ["margin", "vein", "tissue"]
#     blade_msks = [stats.get_layer(multimsk, msk_col_dct, substrct) for substrct in blade_substructs]
#     blade_msk = np.array(blade_msks).sum(axis=0) > 0
    
#     # Review blade mask and potentially redraw leaf contours
#     contours = segmentation.contouring(blade_msk.astype("uint8"))
#     accepted_contours = []
#     for cnt in contours:
#         cnt_img = cv2.drawContours(img.copy(), [cnt], 0, (0,255,0), 1)
#         viz.plot_big(cnt_img)
#         answer = input("Accept contour? (any key/n/skip): ")
#         if answer == "skip":
#             continue
#         elif answer == "n":
#             status = "pending"
#             while status != "done":
#                 # Draw polygon
#                 pdrawer = draw.PolygonDrawer("draw polygon", cnt_img)
#                 polygon = pdrawer.run()
                
#                 # Crop blade mask with hand-drawn polygon
#                 binary_mask = np.zeros(np.shape(img), dtype=np.uint8)
#                 polygon_msk = cv2.drawContours(binary_mask, [polygon], 0, (255,255,255), -1) #Check if indeed this draws a mask
#                 manual_blade_msk = (polygon_msk > 0) & (blade_msk > 0 )
#                 bl_msk_cnt = segmentation.contouring(manual_blade_msk.astype("uint8"))[0]
#                 new_cnt_img = cv2.drawContours(img.copy(), [bl_msk_cnt], 0, (0,255,0), 1)
#                 viz.plot_big(new_cnt_img)
                
#                 # Accept new mask and annotate leaf
#                 if not input("accept new polygon? (y/n)") == "n":
#                     answer = input("leaf type? ")
#                     assert answer in leaf_types, "{} is not an accepted leaf type".format(answer)
#                     assert len(leaf_polygon_dct[fn][answer]) <= 2, "Trying to save >2 polygons for fn: {}, leaf type: {}".format(fn, answer)
#                     leaf_polygon_dct[fn][answer].append(bl_msk_cnt)
#                     with open(LEAFPOLY_DCT_PATH, "wb") as f:
#                         pickle.dump(leaf_polygon_dct, f)
#                 status = input("done? ")
#         else:
#             # Annotate leaf
#             answer = input("leaf type? ")
#             assert answer in leaf_types, "{} is not an accepted leaf type".format(answer)
#             assert len(leaf_polygon_dct[fn][answer]) <= 2, "Trying to save >2 polygons for fn: {}, leaf type: {}".format(fn, answer)
#             leaf_polygon_dct[fn][answer].append(cnt)
#             with open(LEAFPOLY_DCT_PATH, "wb") as f:
#                 pickle.dump(leaf_polygon_dct, f)
#     leaf_polygon_dct[fn]["status"] = "done"
#     with open(LEAFPOLY_DCT_PATH, "wb") as f:
#         pickle.dump(leaf_polygon_dct, f)
        
# # %% Create leaf age multimask
# # TODO write as function, move to segmentation.py

# with open(LEAFPOLY_DCT_PATH, "rb") as f:
#     leaf_polygon_dct = pickle.load(f)

# for fn in leaf_polygon_dct.keys():
#     # Load leaf image and create empty leaf_multimsk
#     img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
#     assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
#     leaf_multimsk = np.zeros((img.shape[0], img.shape[1], 3))

#     # Iterate over leaf classes in random order and assign color to empty image
#     for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
#         for polygon in leaf_polygon_dct[fn][leaf_class]:
#             cv2.drawContours(leaf_multimsk, [polygon], 0, leafmsk_col_dct[leaf_class], -1)
    
#     cv2.imwrite(LEAF_MULTIMSK_PATH + fn, leaf_multimsk.astype("uint8"))


# # %% Output annotated leaf examples
# LEAF_EXAMPLE_PATH = "data/output/leaf_msk_examples/" 

# with open(LEAFPOLY_DCT_PATH, "rb") as f:
#         leaf_polygon_dct = pickle.load(f)
        
# if not os.path.exists(LEAF_EXAMPLE_PATH):
#     os.mkdir(LEAF_EXAMPLE_PATH)
        
# # For img, polygon, find max. coord (bottom-right) print name on picture
# for fn in leaf_polygon_dct.keys():
#     # Load leaf image and create empty leaf_multimsk
#     img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
#     assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    
#     canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
#         for polygon in leaf_polygon_dct[fn][leaf_class]:
#             cv2.drawContours(canvas, [polygon], 0, leafmsk_col_dct[leaf_class], 3)
            
#     for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
#         for polygon in leaf_polygon_dct[fn][leaf_class]:
#             max_x = polygon[:,:,0].min()
#             max_y = polygon[:,:,1].min()
#             cv2.putText(canvas, leaf_class, (max_x,max_y), cv2.FONT_HERSHEY_SIMPLEX,
#                         .5,(255,255,255), 2)
    
#     cv2.imwrite(LEAF_EXAMPLE_PATH + fn, canvas)
    
    
# # Store these in a separate folder, treat like substructures and calculate CQ per substructure

# # Finally, we want some automated way for recognizing these leaves. We could do this two ways:
#     #a. Create a df (per leaf) with a set of numerical features that defines the leaf type / plant and train a NN or SVM on that
#         # Features; leaf_size / plant_size, number of leaves on plant, check some described papers
#     #b. Train a conv. NN on the leaf images

