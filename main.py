# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:40:58 2020

@author: lucas
"""
from scripts import utils, processing, draw, stats
from scripts.viz import *
import itertools

import cv2
import numpy as np
import pandas as pd
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set(font_scale=1.3)

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

DF_SAVE_PATH = "data/Noccaea_processed.csv"
POLY_DCT_PATH = "data/polygon_dict.pck"
LEAFPOLY_DCT_PATH = "data/leaf_polygon_dct.pck"

batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)
plant_fns = os.listdir(PLANT_MSK_PATH)
rand_plant_fns = plant_fns.copy()
random.shuffle(rand_plant_fns)


obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
msk_col_dct = get_colors(obj_class_lst, "Set2")
msk_hex_palette = sns.color_palette(['#%02x%02x%02x' % (msk_col_dct[key][2], msk_col_dct[key][1], msk_col_dct[key][0]) \
                                     for key in obj_class_lst]) # BGR -> RGB -> HEX -> sns palette
msk_hex_palette = dict(zip(obj_class_lst, msk_hex_palette))




leaf_types = ["first", "grown_1", "grown_2", "developping"]
leafmsk_col_dct = get_colors(leaf_types, "hls")


hex_msk_col_dct = {k:'#{:02x}{:02x}{:02x}'.format(v[2],v[1],v[0]) for k,v in msk_col_dct.items()}
RGB_df = pd.Series(hex_msk_col_dct)
RGB_df.to_csv("data/RGB_df.csv")

# %% clear all previous generated

# Re-create polygon dict and empty directories
# polygon_dct = {batch: {} for batch in batchname_lst}
# with open(POLY_DCT_PATH, "wb") as f:
#             pickle.dump(polygon_dct, f)

# fn_lst = os.listdir(PLANT_IMG_PATH)
# for fn in fn_lst:
#     os.unlink(PLANT_IMG_PATH + fn)
#     os.unlink(PLANT_MSK_PATH + fn)
#     os.unlink(PLANT_MULTIMSK_PATH + fn)
#     os.unlink(PLANT_ZIMG_PATH + fn)

#TODO Change all occurences of 'batch' to remove spaces?
# %% 0. Manually divide batches into plants and annotate
df = pd.read_csv(DF_SAVE_PATH, index_col=0)
with open(POLY_DCT_PATH, "rb") as f:
    polygon_dct = pickle.load(f)

# TODO skipped batch 1
# TODO batch 1 and 10 and more ? plants overlap with borders problematic
for batch in batchname_lst:
    img = cv2.imread(RAW_TIFF_PATH + batch + "- " + "Image.tif", cv2.IMREAD_GRAYSCALE)
    all_plants_seen = False
    
    while not all_plants_seen:
        answer = input("Done drawing batch {}? (y)".format(batch))
        if answer == "y":
            all_plants_seen = True
            continue
        
        # Manually draw polygon around plant
        pdrawer = draw.PolygonDrawer("draw polygon", img)
        polygon = pdrawer.run()
                
        # Annotate
        img_poly = cv2.polylines(np.copy(img), polygon, True, (255,255,255), 1)
        plot_big(img_poly)
        accession, replicate = draw.annotate(batch)
        fn = "_".join([batch, accession, replicate])
        fn += ".tif"

        #TODO check whether accession and replicate were already entered?

        # Store polygon and annotation
        polygon_dct[batch][accession + "_" + str(replicate)] = polygon
        with open(POLY_DCT_PATH, "wb") as f:
            pickle.dump(polygon_dct, f)
            
        df.loc[(df["Accession #"] == int(accession)) & (df["Biological replicate"] == replicate) & 
               (df["Plant part"] == "shoot"), ["batch", "fn"]] = [batch, fn]
        df.to_csv(DF_SAVE_PATH)
        
# %% Manually annotate leaf age

# We want a multi-mask again per plant holds classes: background, "first leaf", "developing leaf", "developed leaf 1", "developed leaf 2"
with open(LEAFPOLY_DCT_PATH, "rb") as f:
    leaf_polygon_dct = pickle.load(f)

for fn in rand_plant_fns:

    print("working on: ", fn)
    if fn not in leaf_polygon_dct:
        leaf_polygon_dct[fn] = {leaftype:[] for leaftype in leaf_types}
        leaf_polygon_dct[fn]["status"] = "pending"
    elif leaf_polygon_dct[fn]["status"] == "pending":
        for k,v in leaf_polygon_dct[fn].items():
            print(k, ": ", len(v))
    elif leaf_polygon_dct[fn]["status"] == "done":
        continue
    else:
        print("Something weird is happening")      
        
        
    img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
    assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
    
    # Find existing blade mask
    blade_substructs = ["margin", "vein", "tissue"]
    blade_msks = [stats.get_layer(multimsk, msk_col_dct, substrct) for substrct in blade_substructs]
    blade_msk = np.array(blade_msks).sum(axis=0) > 0
    
    # Review and potentially redraw leaf contours
    contours = processing.contouring(blade_msk.astype("uint8"))
    accepted_contours = []
    for cnt in contours:
        cnt_img = cv2.drawContours(img.copy(), [cnt], 0, (0,255,0), 1)
        plot_big(cnt_img)
        answer = input("Accept contour? (y/ENTER/n/skip): ")
        if answer == "skip":
            continue
        elif answer == "n":
            status = "pending"
            while status != "done":
                # Draw polygon
                pdrawer = draw.PolygonDrawer("draw polygon", cnt_img)
                polygon = pdrawer.run()
                
                # Crop blade mask with hand-drawn polygon
                binary_mask = np.zeros(np.shape(img), dtype=np.uint8)
                polygon_msk = cv2.drawContours(binary_mask, [polygon], 0, (255,255,255), -1) #Check if indeed this draws a mask
                manual_blade_msk = (polygon_msk > 0) & (blade_msk > 0 )
                bl_msk_cnt = processing.contouring(manual_blade_msk.astype("uint8"))[0]
                new_cnt_img = cv2.drawContours(img.copy(), [bl_msk_cnt], 0, (0,255,0), 1)
                plot_big(new_cnt_img)
                if not input("accept new polygon? (y/n)") == "n":
                    answer = input("leaf type? ")
                    assert answer in leaf_types, "{} is not an accepted leaf type".format(answer)
                    assert len(leaf_polygon_dct[fn][answer]) <= 2, "Trying to save >2 polygons for fn: {}, leaf type: {}".format(fn, answer)
                    leaf_polygon_dct[fn][answer].append(bl_msk_cnt)
                    with open(LEAFPOLY_DCT_PATH, "wb") as f:
                        pickle.dump(leaf_polygon_dct, f)
                status = input("done? ")
        else:
            answer = input("leaf type? ")
            assert answer in leaf_types, "{} is not an accepted leaf type".format(answer)
            assert len(leaf_polygon_dct[fn][answer]) <= 2, "Trying to save >2 polygons for fn: {}, leaf type: {}".format(fn, answer)
            leaf_polygon_dct[fn][answer].append(cnt)
            with open(LEAFPOLY_DCT_PATH, "wb") as f:
                pickle.dump(leaf_polygon_dct, f)
    leaf_polygon_dct[fn]["status"] = "done"
    with open(LEAFPOLY_DCT_PATH, "wb") as f:
        pickle.dump(leaf_polygon_dct, f)
        
# %% Create leaf multimask
with open(LEAFPOLY_DCT_PATH, "rb") as f:
    leaf_polygon_dct = pickle.load(f)

for fn in leaf_polygon_dct.keys():
    # Load leaf image and create empty leaf_multimsk
    img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    leaf_multimsk = np.zeros((img.shape[0], img.shape[1], 3))

    # Iterate over leaf classes in random order and assign color to empty image
    for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
        for polygon in leaf_polygon_dct[fn][leaf_class]:
            cv2.drawContours(leaf_multimsk, [polygon], 0, leafmsk_col_dct[leaf_class], -1)
    
    cv2.imwrite(LEAF_MULTIMSK_PATH + fn, leaf_multimsk.astype("uint8"))


# %% Output annotated leaf examples

# For img, polygon, find max. coord (bottom-right) print name on picture
for fn in leaf_polygon_dct.keys():
    # Load leaf image and create empty leaf_multimsk
    img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
        for polygon in leaf_polygon_dct[fn][leaf_class]:
            cv2.drawContours(canvas, [polygon], 0, leafmsk_col_dct[leaf_class], 3)
            
    for leaf_class in random.sample(leaf_types, (len(leaf_types))) : # Randomizing because some leaf masks overlap, leaf class coming out on top is random
        for polygon in leaf_polygon_dct[fn][leaf_class]:
            max_x = polygon[:,:,0].min()
            max_y = polygon[:,:,1].min()
            cv2.putText(canvas, leaf_class, (max_x,max_y), cv2.FONT_HERSHEY_SIMPLEX,
                        .5,(255,255,255), 2)
    
    cv2.imwrite("data/output/leaf_msk_examples/" + fn, canvas)
    
    
# Store these in a separate folder, treat like substructures and calculate CQ per substructure

# Finally, we want some automated way for recognizing these leaves. We could do this two ways:
    #a. Create a df (per leaf) with a set of numerical features that defines the leaf type / plant and train a NN or SVM on that
        # Features; leaf_size / plant_size, number of leaves on plant, check some described papers
    #b. Train a conv. NN on the leaf images

# %% 1. Create batch foreground masks
layers = ["Ca.tif", "K.tif", "Ni.tif",] # "Image.tif"
for batch in batchname_lst:
    binary_mask_lst = []
    for layer in layers:
        img = cv2.imread(RAW_TIFF_PATH + batch + "- " + layer, 0)
        ret,th_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # plot_big(th_img)
        # con = processing.contouring(th_img)
        # binary_mask_lst.append(processing.create_mask(img, con))
        binary_mask_lst.append(th_img)
        
    # Take union over batch masks # TODO change to cv2.add
    mask = (binary_mask_lst[0] == 255) | (binary_mask_lst[1] == 255) | (binary_mask_lst[2] == 255) #\
            #| (binary_mask_lst[3] == 255)
    mask = (mask * 255).astype("uint8")
    
    # Remove noise
    noise_kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)
    close_kernel = np.ones((3,3), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    
    # plot_big2(mask, mask_closed, batch)
    cv2.imwrite(BATCH_MSK_PATH + batch + "batchmsk.tif", mask_closed)
    
# %% 2. Create batch multimask
for batch in batchname_lst:
    img_fn = RAW_TIFF_PATH + batch + "- Image.tif"
    img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(img_fn)

    msk_fn = BATCH_MSK_PATH + batch + "batchmsk.tif"
    msk = cv2.imread(msk_fn, cv2.IMREAD_GRAYSCALE)
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(msk_fn)
    
    msk_dct = {}
    
    # Get background
    background = np.where(msk == 0, 255, 0)
    msk_dct["background"] = background
    
    # Get blade by opening on the whole plant mask with a large kernel to remove the petiole and artefacts
    blade_kernel = np.ones((15,15),np.uint8)
    blade = cv2.morphologyEx(msk, cv2.MORPH_OPEN, blade_kernel)
    blade = np.where((blade == 255) & (msk == 255), 255, 0) # Opening adds some pixels outside mask I beleive
    
    
    # Now get the petiole masks by subtracting the blade from the whole plant mask 
    # followed by another smaller kernel opening
    petiole = ((msk != blade) * 255).astype("uint8")
    large_contours = processing.contouring(petiole, area_th = 0.00001) # Removes small misclassified petiole areas at blade edge
    petiole = processing.create_mask(petiole, large_contours)
    petiole = (((petiole == 255) & (background == 0)) * 255).astype("uint8") # Removes artefacts created during contouring
    msk_dct["petiole"] = petiole
    
    # Assign blade + all unassigned pixels to blade
    blade = ((background + petiole) == 0) * 255
    blade = blade.astype("uint8")
    
    ## Get leaf margin
    margin_kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(blade, cv2.MORPH_GRADIENT, margin_kernel)
    margin = np.where((blade == 255) & (gradient == 255), 255, 0).astype("uint8")
    msk_dct["margin"] = margin

    ## Get vein mask
    blade_img = np.where(blade, img, 0)
    lap_img = cv2.Laplacian(blade_img,cv2.CV_64F, ksize=7)
    
    thin_veins = (lap_img < -2500) * 255 #np.where(lap_img < -2500, 255, 0)
    fat_veins = (lap_img < 0) * 255
    skeleton_veins = cv2.ximgproc.thinning(fat_veins.astype("uint8"))
    
    veins = cv2.add(skeleton_veins,thin_veins.astype("uint8") ) 
    marginless_veins = np.where((veins == 255) & (blade == 255) & (margin == 0), 255, 0)
    msk_dct['vein'] = marginless_veins
    
    ## get leaf tissue 
    # TODO remove margin == 0 shouldn't matter
    tissue = np.where((marginless_veins == 0) & (blade == 255) & (margin == 0), 255, 0)
    msk_dct["tissue"] = tissue
    
    ## Check for overlap between masks
    for tup in itertools.combinations(msk_dct, 2):
        msk0 = msk_dct[tup[0]]
        msk1 = msk_dct[tup[1]]
        overlap = (msk0 == 255) & (msk1 == 255)
        if overlap.any() == True:
            print("'{} and {}' overlap".format(tup[0], tup[1]))
            # plot_big(np.where(overlap, (255,255,255), (0,0,0)))
            plot_big(overlap * 255)
            assert overlap.any() == False, "'{} and {}' overlap".format(tup[0], tup[1])

    ## Create multi-color mask image (.jpg)
    multi_msk = np.zeros((msk.shape[0], msk.shape[1], 3))
    for name, partial_msk in msk_dct.items():
        col_BGR = msk_col_dct[name]
        partial_msk = partial_msk[:,:,None] # Add dimension for color
        multi_msk = np.where(partial_msk == 255, col_BGR, multi_msk)
    
    cv2.imwrite(BATCH_MULTIMSK_PATH + batch + "multimsk.tif",  multi_msk.astype("uint8"))


# %% Create individual images per plant
metals = ["Z"]

for metal in metals:
    processing.make_individual_plant_images(POLY_DCT_PATH, batchname_lst, RAW_TIFF_PATH, 
                                 BATCH_MSK_PATH, BATCH_MULTIMSK_PATH, RAW_TXT_PATH, 
                                 PLANT_IMG_PATH, PLANT_MSK_PATH, PLANT_MULTIMSK_PATH,
                                 metal, msk_col_dct, create_masks = True)

# %% Get stats from image
metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]
substructures = obj_class_lst[1:] + ["plant", "rand_5", "rand_10"]
df = pd.read_csv("data/Noccaea_nometrics.csv", index_col=0)

plant_fns = os.listdir(PLANT_MSK_PATH)
# plant loop
for fn in plant_fns:
    # load plant mask and multi-mask
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
    
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
       
    # metals loop
    for metal in metals:
        # load metal image
        metal_name = metal.split("_")[1]
        METAL_PATH = "data/plant_" + metal_name + "img/"
        img = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
    
        # TODO noise levels loop (if still relevant)
        # substructure loop
        for substrct in substructures:
           # load correct mask
           if substrct == "plant":
               layer_msk = msk
           elif substrct == "rand_5":
               layer_msk = cv2.imread(PLANT_RANDMSK_PATH + "5/" + fn, cv2.IMREAD_GRAYSCALE)
           elif substrct == "rand_10":
               layer_msk = cv2.imread(PLANT_RANDMSK_PATH + "10/" + fn, cv2.IMREAD_GRAYSCALE)
           elif substrct in obj_class_lst[1:]:
               layer_msk = stats.get_layer(multimsk, msk_col_dct, substrct)
           else:
               raise Exception("substructure: " + substrct + " is invalid")
           subs_metal_image = stats.get_sub_ele_img(img, layer_msk)
           abs_metal, n_pixels, meanC = stats.get_sub_ele_stats(subs_metal_image)
           A500 = stats.XrandPixel_value(layer_msk, img, fn, substrct, 500)
           colnames = ["_".join((metal, substrct, metric)) for metric in ["abs", "n_pix", "meanC", "A500"]]
           df.loc[df["fn"] == fn, colnames] = [abs_metal, n_pixels, meanC, A500]

        # Calculate CQ for all substructures
        CQ_colnames = ["_".join((metal, substrct, "CQ")) for substrct in substructures]
        mean_colnames = ["_".join((metal, substrct, "meanC")) for substrct in substructures]
        plant_mean_colname = "_".join((metal, "plant", "meanC"))
        df[CQ_colnames] = df[mean_colnames].div(df[plant_mean_colname], axis=0)
df.to_csv("data/Noccaea_CQsA500.csv")

# %% Find units for "absolute"
df = pd.read_csv("data/Noccaea_CQsA500.csv")

df["ICP:muXRF_ratio"] = df["Zn"] / df["metal_Z_plant_abs"]
plt.scatter(df.index, df["ICP:muXRF_ratio"])
plt.title("measured Zinc : machine vision zinc ratio")
plt.show()

sns.scatterplot(x="metal_Z_plant_n_pix", y = "ICP:muXRF_ratio", data=df)
plt.ylabel("ICP-AES : muXRF ratio")
plt.xlabel("plant size [pixels]")
plt.savefig("data/output/results/ICP-AES_muXRF ratio_plantsize_cor.png", bbox_inches="tight")
plt.show()

sns.scatterplot(x="metal_Z_plant_meanC", y = "ICP:muXRF_ratio", data=df)
plt.ylabel("ICP-AES : muXRF ratio")
plt.xlabel("mean Zinc concentration")
plt.show()

metal_name = "Z"
METAL_PATH = "data/plant_" + metal_name + "img/"

for fn in outlier_abs_fn:
    Zn_image = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
    plot_big(Zn_image)
    break

# %% Correlate metals



# %% Calculate H2
# pymer is very unstable

os.environ["R_HOME"] = r"C:/Program Files/R/R-3.6.3"
os.environ["PATH"]   = r"C:/Program Files/R/R-3.6.3/bin/x64" + ";" + os.environ["PATH"]
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pymer4.models import Lmer


df = pd.read_csv("data/Noccaea_CQsA500.csv")

phenotypes = ["plant_n_pix","plant_meanC", "petiole", "margin", "vein", "tissue", "rand_5", "rand_10"]
metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]
metric = "CQ"

for metal in metals:
    for phenotype in phenotypes:
        if (phenotype == "plant_n_pix") or (phenotype == "plant_meanC"):
            colname = "_".join(metal, phenotype)
        else:
            colname = "_".join(metal, phenotype, metric)
        
        fitlmer_rand = Lmer(colname + " ~ (1|Accession #)",  data=df, REML=True)
        
        # Lmer("DV ~ IV2 + (IV2|Group)", data=df)
        break

# %% Show raw data that supports H2
random.seed(69)
df = pd.read_csv("data/Noccaea_CQsA500.csv")
sns.set_style("ticks")

substructure = "petiole"
metal = "metal_Z"
metric = "CQ"
colname = "_".join((metal, substructure, metric))

df = df.loc[df["batch"].notna(),:]
random_accessions = random.sample(df["Accession #"].unique().tolist(), 10)
random_accessions.sort()
acc_strs = [str(x) for x in random_accessions]

plt_df = df.loc[df["Accession #"].isin(random_accessions),["Accession #", colname]]
plt_df["substructure"] = [substructure] * len(plt_df)


sns.catplot(x="Accession #", y=colname, data=plt_df, hue="substructure", palette=msk_hex_palette, legend=False)
plt.ylabel("Petiole Zinc CQ [-]")
plt.savefig("data/output/results/randCQ_petiole_Zn.png", bbox_inches="tight")
plt.show()

# %% Scatter absolute metal concentration against plant size and mean zinc concentration
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df["accession_str"] = df['Accession #'].astype(str)

metal = "metal_Z"
metric = "abs"
npix = "_".join((metal, "plant", "n_pix"))

sns.set_style("ticks")

subs_abs = [ "_".join((metal, substr, metric)) for substr in obj_class_lst[1:] ]
plt.figure(figsize=(10,10))
for i,subs in enumerate(subs_abs):
    plt.subplot(2,2,i+1)
    df_plot = df.loc[:,[npix,subs]]
    df_plot.loc[:,"substructure"] = [subs.split("_")[2]] * len(df_plot)
    sns.scatterplot(x=npix, y = subs, data=df_plot, hue = "substructure", palette=msk_hex_palette, legend=False)
    # plt.title(subs)
    plt.xlabel("plant size [pixels]")
    plt.ylabel("absolute zinc concentration []")
plt.savefig("data/output/results/abs_size_cor.png")
plt.show()

meanC = "_".join((metal, "plant", "meanC"))
subs_abs = [ "_".join((metal, substr, metric)) for substr in obj_class_lst[1:] ]
plt.figure(figsize=(10,10))
for i,subs in enumerate(subs_abs):
    plt.subplot(2,2,i+1)
    df_plot = df.loc[:,[meanC,subs]]
    df_plot.loc[:,"substructure"] = [subs.split("_")[2]] * len(df_plot)
    sns.scatterplot(x=meanC, y = subs, data=df_plot, hue = "substructure", palette=msk_hex_palette, legend=False)
    # plt.title(subs)
    plt.xlabel("mean Zinc concentration [-]")
    plt.ylabel("absolute zinc concentration []")
plt.savefig("data/output/results/meanC_size_cor.png")
plt.show()

              
# %% Scatter CQs against n_pixel and mean plant CQ
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df["accession_str"] = df['Accession #'].astype(str)


subs_CQ = [ name + "_CQ" for name in obj_class_lst[1:] ]
plt.figure(figsize=(10,10))
for i,subs in enumerate(subs_CQ):
    plt.subplot(2,2,i+1)
    plt.scatter(df["plant_npixel"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant size [pixels]")
    plt.ylabel("CQ")
plt.savefig("data/output/plots/CQ versus plant size.png")
plt.show()
    

plt.figure(figsize=(10,10))
for i, subs in enumerate(subs_CQ):
    plt.subplot(2,2, i+1)
    plt.scatter(df["plant_meanZC"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant mean Zinc concentration")
    plt.ylabel("CQ")
plt.savefig("data/output/plots/CQ versus mean Zinc concentration.png")
plt.show()

# Adapt CQ by subtracting 1 and taking absolute value
normean_abs_CQs = [name + "normean_abs_CQ" for name in obj_class_lst[1:]]
CQs = [subs + "_CQ" for subs in obj_class_lst[1:]]
df[normean_abs_CQs] = (df[CQs] - 1).abs()

plt.figure(figsize=(10,10))
for i, subs in enumerate(normean_abs_CQs):
    plt.subplot(2,2, i+1)
    plt.scatter(df["plant_meanZC"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant mean Zinc concentration")
    plt.ylabel("mean normalized absolute CQ")
plt.savefig("data/output/plots/mean-normalized absolute CQ versus mean Zinc concentration.png")
plt.show()

# %% scatter CQs against each other
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df["accession_str"] = df['Accession #'].astype(str)
pairplt_vars = [ name + "_CQ" for name in obj_class_lst[1:] ] + ['accession_str']
df_noNAN = df.loc[df['batch'].notna(),:]

from scipy.stats import pearsonr
def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

g = sns.pairplot(df_noNAN[pairplt_vars]) # , hue='accession_str', palette='bright'
g.map(corrfunc)
# g._legend.remove()

# %% Scatter CQ against relative area proportion of substructure
df = pd.read_csv("data/Noccaea_CQsA500.csv")
substructures = obj_class_lst[1:]
metal = "metal_Z"
plt.figure(figsize=(10,10))
for i, substrct in enumerate(substructures):
    sbstrct_area = "_".join((metal, substrct, "n_pix"))
    plant_area = "_".join((metal, "plant", "n_pix"))
    rel_strct_area = df[sbstrct_area] / df[plant_area] * 100
    substrct_CQ = "_".join((metal, substrct, "CQ"))
    
    plt.subplot(2,2,i+1)
    plt.scatter(rel_strct_area, df[substrct_CQ], s=2)
    plt.xlabel("relative substructure area [%]")
    plt.ylabel("CQ")
    plt.title(substrct)
plt.savefig("data/output/results/CQ versus rel subs area.png", bbox_inches="tight")
plt.show()


# %% Check whether means of subs sum up to mean of plant
# CQ_cols =  [ name + "_CQ" for name in obj_class_lst[1:] ]
# df["CQ_sum"] = df[CQ_cols].sum(axis=1)
# plt.boxplot(df.loc[df['batch'].notna(), "CQ_sum"])

mean_cols =  [ name + "_meanZC" for name in obj_class_lst[1:] ]
npixel_cols = [ name + "_npixel" for name in obj_class_lst[1:] ]
weighted_means = [ name + "_weightedZC" for name in obj_class_lst[1:] ]

# Check if n_pixels add up
subs_npixel = df[npixel_cols].sum(axis=1)
df["diff_npixel"] = df["plant_npixel"] - subs_npixel
plt.boxplot(df.loc[df['batch'].notna(), "diff_npixel"])

# weighted_means
df[weighted_means] = df[mean_cols] * df[npixel_cols]
df["plant_meanZC_check"] = (df[weighted_means].sum(axis=1) / df['plant_npixel']) - df['plant_meanZC']
plt.boxplot(df.loc[df['batch'].notna(), "plant_meanZC_check"])


# df["mean_of_means"] = df[mean_cols].sum(axis=1) / len(mean_cols)


# %% check if all mask pixels are assigned to a multimask layer

# %% create mask r-cnn dictionary and images
# TODO first got id "0" this should be reserved for bg right?

from skimage import io
from skimage import color
from seaborn import color_palette

# Create images from polygons
with open(LEAFPOLY_DCT_PATH, "rb") as f:
    leaf_polydct = pickle.load(f)


# Create forward leaf type color dct
leaftypes_rep2 = []
for leaftype in leaf_types:
    leaftypes_rep2.extend([leaftype, leaftype])

cols_float = color_palette("hls", 8)
cols_int = []
for col in cols_float:
    col = tuple([int(x * 255) for x in col])
    cols_int.append(col)

fwd_coldct = {'first':[cols_int[0], cols_int[1]], 'grown_1':[cols_int[2], cols_int[3]],
              'grown_2':[cols_int[4],cols_int[5]], 'developping':[cols_int[6],cols_int[7]]}
bwd_coldct = {cols_int[0]:('first', 1), cols_int[1]:('first', 1), cols_int[2]:('grown_1', 2),
              cols_int[3]:('grown_1', 2), cols_int[4]:('grown_2', 3), cols_int[5]:('grown_2', 3),
              cols_int[6]:('developping', 4),cols_int[7]:('developping', 4)
    }


with open("data/output/ML_imgs/col_classid_dct.pck", "wb") as f:
    pickle.dump(bwd_coldct, f)

ML_msk_path =  "data/output/ML_imgs/mask/"
ML_img_path = "data/output/ML_imgs/image/"

for fn in leaf_polydct.keys():
    plant_img = io.imread(PLANT_IMG_PATH + fn)
    img_rgb = color.gray2rgb(plant_img)

    mrcnn_msk = np.zeros((plant_img.shape[0], plant_img.shape[1], 3))
    mrcnn_msk_cop = mrcnn_msk.copy()
    
    for leaftype in leaf_types: # Randomizing because some leaf masks overlap, leaf class coming out on top is random
        for i,polygon in enumerate(leaf_polygon_dct[fn][leaftype]):
            cv2.drawContours(mrcnn_msk, [polygon], 0, fwd_coldct[leaftype][i], -1)
    # plt.subplot(2,1,1)
    # plt.imshow(mrcnn_msk.astype("int32"))
    # plt.subplot(2,1,2)
    # plt.imshow(img_rgb)
    # break
    io.imsave(ML_msk_path + fn.split(".")[0] + "_label.png", mrcnn_msk.astype("uint8"))
    io.imsave(ML_img_path + fn.split(".")[0] + "_rgb.png", img_rgb)
    

mask = []
for fn in leaf_polydct.keys():
    img = io.imread(ML_msk_path + fn.split(".")[0] + "_label.png")
    img_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0, return_inverse=True)
    background = np.array([0,0,0]) # Background is black
    class_id_lst = []
    for i, color in enumerate(img_colors[0]):
        if (color != background).any():
            bin_mask = np.where(img_colors[1] == i, True, False).astype(int)
            bin_mask = np.reshape(bin_mask, img.shape[:2]) # Shape back to 2D
            class_name = bwd_coldct[tuple(color)][0]
            class_id = bwd_coldct[tuple(color)][1]
            plt.imshow(bin_mask)
            plt.title(class_name + " " + str(class_id))
            plt.show()
            mask.append(bin_mask)
    import pdb; pdb.set_trace()
        

# %% copy test val to right folders
from sklearn.model_selection import train_test_split

ML_msk_path =  "data/output/ML_imgs/mask/"
ML_img_path = "data/output/ML_imgs/image/"

train, test = train_test_split(os.listdir(ML_img_path))

for fn in train:
    fn_bare = fn[:-7]
    fn_mask = fn_bare + "label.png"
    
    
    
    
    