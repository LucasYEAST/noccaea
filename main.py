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
from scipy.stats import pearsonr
# sns.set_style("ticks")
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

DF_SAVE_PATH = "data/Noccaea_CQsA500.csv"
POLY_DCT_PATH = "data/polygon_dict.pck"
LEAFPOLY_DCT_PATH = "data/leaf_polygon_dct.pck"

batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)
plant_fns = os.listdir(PLANT_MSK_PATH)
rand_plant_fns = plant_fns.copy()
random.shuffle(rand_plant_fns)


obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
class_dct = {1:"petiole", 2:"margin", 3:"vein", 4:"tissue"}
class_dct_rev = {v:k for k,v in class_dct.items()}

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
# TODO write as function, move to processing.py

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
    blade_kernel = np.ones((blade_ksize,blade_ksize),np.uint8)
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
    lap_img = cv2.Laplacian(blade_img,cv2.CV_64F, ksize=lap_ksize)
    
    thin_veins = (lap_img < thin_th) * 255 #np.where(lap_img < -2500, 255, 0)
    fat_veins = (lap_img < fat_th) * 255
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
            viz.plot_big(overlap * 255)
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
    processing.make_individual_plant_images(POLY_DCT_PATH, batchname_lst[0], RAW_TIFF_PATH, 
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
from scipy.stats.stats import pearsonr

df = pd.read_csv("data/Noccaea_CQsA500.csv")
sns.set_style("ticks")

df["ICP:muXRF_ratio"] = df["Zn"] / df["metal_Z_plant_abs"]
outlier_abs_fn = df.loc[df["ICP:muXRF_ratio"] > 0.00075,"fn"].tolist()

plt.scatter(df.index, df["ICP:muXRF_ratio"])
plt.title("measured Zinc : machine vision zinc ratio")
plt.show()

sns.scatterplot(x="metal_Z_plant_n_pix", y = "ICP:muXRF_ratio", data=df)
plt.ylabel("ICP-AES : \u03BCXRF ratio")
plt.xlabel("Plant Size [pixels]")
plt.savefig("data/output/results/ICP-AES_muXRF ratio_plantsize_cor.png", bbox_inches="tight")
plt.show()

sns.scatterplot(x="metal_Z_plant_meanC", y = "ICP:muXRF_ratio", data=df)
plt.ylabel("ICP-AES : muXRF ratio")
plt.xlabel("mean Zinc concentration")
plt.show()


sns.scatterplot(x="Zn", y = "metal_Z_plant_abs", data=df)
dfcor = df.loc[(df.Zn.notna()) & (df.metal_Z_plant_abs.notna()),:]
r,p = pearsonr(dfcor.Zn, dfcor.metal_Z_plant_abs)
print(r,p)
plt.ylabel("\u03BCXRF absolute zinc [-]")
plt.xlabel("ICP-AES absolute zinc [\u03BCm]")
plt.title("r: " + str(r))
plt.show()


metal_name = "Z"
METAL_PATH = "data/plant_" + metal_name + "img/"

# for fn in outlier_abs_fn:
#     Zn_image = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
#     img = cv2.imread(PLANT_IMG_PATH + fn)
#     multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn)
#     msk = cv2.imread(PLANT_MSK_PATH + fn)
#     print("pixel size of mask", (msk / (255 * 3)).sum() )
#     print(df.loc[df.fn == fn, "ICP:muXRF_ratio"])
#     plot_big(Zn_image)
#     # plot_big2(img, msk)
#     plot_big2(img, multimsk)
#     if input("save? ") == "yes":
#         cv2.imwrite("data/output/article_images/wrongclass_smallimg_" + fn + ".png", Zn_image)

# %% Plot mean zinc concentration for random plants
random.seed(69)
sns.set_style("ticks")
df = pd.read_csv("data/Noccaea_CQsA500.csv", index_col=0)
random_accessions = random.sample(list(df["Accession #"].unique()), 15)
random_accessions.sort()
# acc_strs = [str(x) for x in random_accessions]

plt_df = df.loc[df["Accession #"].isin(random_accessions),["Accession #", "metal_Z_plant_meanC"]]
plt.figure(figsize=(7,5))
sns.catplot(x="Accession #", y="metal_Z_plant_meanC", data=plt_df, jitter=False) #hue="Accession #", palette=msk_hex_palette, legend=False
plt.ylabel("mean zinc concentration [-]")
plt.savefig("data/output/article_images/mean_zinc_concentration.png", dpi=300)

# %% Histograms of metals
sns.reset_orig()
metals = ["Zn","Ca","K","Ni"]

for metal in metals:
    metalimg_path = RAW_TXT_PATH + batch + "- " + metal + ".txt"
    img = np.loadtxt(metalimg_path, delimiter=",", skiprows=1).flatten()
    plt.hist(img, bins=256, range=(img.min(), img.max()))   
    # plt.hist(img.ravel(), bins=len(img.ravel())//10)
    plt.show()
    
# %% Correlate metals

metals = ["Z","Ca","K","Ni"]
metal_cor_dct = {metal:np.array([]) for metal in metals}
for fn in plant_fns:   
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255
    for metal in metals:
        METAL_PATH = "data/plant_" + metal + "img/"
        img = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
        img = img[msk == 1]
        img_norm = (img - img.min()) / (img.max() - img.min())
        metal_cor_dct[metal] = np.append(metal_cor_dct[metal], img_norm)

metal_cor_df = pd.DataFrame(metal_cor_dct)
metal_correlations = metal_cor_df.sample(n=100000).corr()
sns.pairplot(metal_cor_df.sample(5000))

# %% Get normalized mean concentration of substructures
sns.set_style("ticks")
mean_subC_dct = {substrct:[] for substrct in obj_class_lst[1:]}
for fn in plant_fns:
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn)
    img = np.genfromtxt("data/plant_Zimg/" + fn.split(".")[0] + ".txt", delimiter=",")
    img_norm = (img - img.mean())/img.std() # Z-normalization
    for substrct,lst in mean_subC_dct.items():
        layer_msk = stats.get_layer(multimsk, msk_col_dct, substrct)
        subs_metal_image = stats.get_sub_ele_img(img_norm, layer_msk)
        _, _, meanC = stats.get_sub_ele_stats(subs_metal_image)
        lst.append(meanC)

mean_subC_df = pd.DataFrame(mean_subC_dct)       
sns.boxplot(data=mean_subC_df, palette=msk_hex_palette)


# %% Check normalized mean concentrations for significant differences
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, kruskal
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols

sns.set_style("ticks")
mean_subC_df = pd.read_csv("data/mean_sub_conc.csv", index_col=0, header=0)
# # Normally d=2)

# Implement non-parametistributed? ANSWER, No
# for substr in mean_subC_df.columns:
#     mean_subC_df[substr].hist(bins=len(mean_subC_df)//2)
#     plt.show()
#     stat, p = shapiro(mean_subC_df[substr])
#     print('Statistics=%.3f, p=%.3f' % (stat, p))
#     # interpret
#     alpha = 0.05
#     if p > alpha:
#     	print('Sample looks Gaussian (fail to reject H0)')
#     else:
#     	print('Sample does not look Gaussian (reject H0)')

# # Significant ANOVA?
# df_melt = pd.melt(mean_subC_df.reset_index(), id_vars=['index'], value_vars=mean_subC_df.columns)
# df_melt.columns = ['index', 'treatments', 'value']
# # Ordinary Least Squares (OLS) model
# model = ols('value ~ C(treatments)', data=df_melt).fit()
# stat, p = shapiro(model.resid)
# alpha = 0.05
# if p > alpha:
#     print('Residuals look Gaussian (fail to reject H0)')
# else:
#     print('Residuals do not look Gaussian (reject H0)')    
# anova_table = sm.stats.anova_lm(model, typeric tests

df = mean_subC_df
stat, p = kruskal(df["petiole"],df['margin'],df['vein'],df['tissue'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

plt.figure(figsize=(7,5))
sns.boxplot(data=mean_subC_df, palette=msk_hex_palette)
plt.ylabel("normalized mean concentration [-]")
plt.savefig("data/output/article_images/normalized_mean_substructure_conc.png", dpi=300)
sp.posthoc_dunn([df["petiole"],df['margin'],df['vein'],df['tissue']], p_adjust = 'bonferroni')

# for p,text in zip(ax.patches, ['a','b','c','d']):
#     height = p.get_height()
#     ax.text(p.get_x()+p.get_width()/2.,
#             height,
#             text,
#             ha="center")
# plt.show()

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
# plt.savefig("data/output/results/abs_size_cor.png")
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
# plt.savefig("data/output/results/meanC_size_cor.png")
plt.show()

              
# %% Scatter CQs against n_pixel and mean plant CQ
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df = df.loc[df.batch.notna(),:]


subs_CQ = [ "metal_Z_" + name + "_CQ" for name in obj_class_lst[1:] ]
plt.figure(figsize=(10,10))
print("Plant size correlation")
for i,subs in enumerate(subs_CQ):
    r,p = pearsonr(df["metal_Z_plant_n_pix"], df[subs])
    print(subs, " r = ", r, " p = ", p)
    plt.subplot(2,2,i+1)
    plt.scatter(df["metal_Z_plant_n_pix"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant size [pixels]")
    plt.ylabel(subs + " CQ")
plt.savefig("data/output/plots/CQ versus plant size.png")
plt.show()

print("\n Mean concentration correlation")
plt.figure(figsize=(10,10))
for i, subs in enumerate(subs_CQ):
    r,p = pearsonr(df["metal_Z_plant_meanC"], df[subs])
    print(subs, " r = ", r, " p = ", p)
    plt.subplot(2,2, i+1)
    plt.scatter(df["metal_Z_plant_meanC"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant mean Zinc concentration")
    plt.ylabel(subs + " CQ")
plt.savefig("data/output/plots/CQ versus mean Zinc concentration.png")
plt.show()

# Adapt CQ by subtracting 1 and taking absolute value
normean_abs_CQs = [name + "normean_abs_CQ" for name in obj_class_lst[1:]]
df[normean_abs_CQs] = (df[subs_CQ] - 1).abs()

plt.figure(figsize=(10,10))
for i, subs in enumerate(normean_abs_CQs):
    plt.subplot(2,2, i+1)
    plt.scatter(df["metal_Z_plant_meanC"], df[subs], s=3)
    plt.title(subs)
    plt.xlabel("plant mean Zinc concentration")
    plt.ylabel("mean normalized absolute CQ")
plt.savefig("data/output/plots/mean-normalized absolute CQ versus mean Zinc concentration.png")
plt.show()

plt.scatter(df["metal_Z_plant_meanC"], df["metal_Z_plant_n_pix"])
plt.title('scatter of mean zinc versus plant size')


# %% scatter CQs against each other
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df["accession_str"] = df['Accession #'].astype(str)
df_noNAN = df.loc[df['batch'].notna(),:]

def return_pval(x,y):
    return pearsonr(x, y)[1]


substructures = obj_class_lst[1:]
metal = "metal_Z"

pairplt_vars = [ "_".join((metal, name, "CQ")) for name in obj_class_lst[1:]]
CQ_pair_corrs = df_noNAN[pairplt_vars].corr(method='pearson')
CQ_pair_corrs.columns = substructures
CQ_pair_corrs.index = substructures
sns.heatmap(CQ_pair_corrs, annot=True, cbar=False, cmap="viridis")
plt.savefig("data/output/article_images/CQ_paircorr.png", dpi=300)

CQ_pair_cor_pval = df_noNAN[pairplt_vars].corr(method=return_pval)
g = sns.pairplot(df_noNAN[pairplt_vars]) # , hue='accession_str', palette='bright'
# g._legend.remove()

# %% Scatter CQ against relative area proportion of substructure
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df = df.loc[df.batch.notna(),:]


substructures = obj_class_lst[1:]
metal = "metal_Z"
plt.figure(figsize=(10,10))
print("Correlation of CQ to relative structure area")
for i, substrct in enumerate(substructures):
    
    sbstrct_area = "_".join((metal, substrct, "n_pix"))
    plant_area = "_".join((metal, "plant", "n_pix"))
    rel_strct_area = df[sbstrct_area] / df[plant_area] * 100
    substrct_CQ = "_".join((metal, substrct, "CQ"))
    
    r,p = pearsonr(rel_strct_area, df[substrct_CQ])
    print("mean substructure area for: ", substrct, " = ", rel_strct_area.mean())
    print(substrct, " r = ", r, " p = ", p)
    
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
    
    
    
    
# %% Find smallest, medium and largest plant
df = pd.read_csv("data/Noccaea_CQsA500.csv")
sorted_df = df.loc[df["batch"].notna(),:].sort_values(by="metal_Z_plant_n_pix")
biggest_plant = sorted_df.fn.iloc[-1]
smallest_plant = sorted_df.fn.iloc[0]
medium_plant = sorted_df.fn.iloc[(len(df) - 1)//2]

# for fn in [biggest_plant, medium_plant, smallest_plant]:
#     img = cv2.imread(PLANT_IMG_PATH + fn, cv2.IMREAD_GRAYSCALE)
#     assert isinstance(img, np.ndarray), "{} doesn't exsit".format(fn)
    
#     multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
#     assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)

#     cv2.imwrite("data/output/curated_masks/img_" + fn, img)
#     cv2.imwrite("data/output/curated_masks/original_multimsk_" + fn, multimsk)

img = cv2.imread(PLANT_IMG_PATH + biggest_plant, cv2.IMREAD_GRAYSCALE)
assert isinstance(img, np.ndarray), "{} doesn't exsit".format(biggest_plant)

multimsk = cv2.imread(PLANT_MULTIMSK_PATH + biggest_plant) # Load as RGB
assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + biggest_plant)

bigsquare_multimsk = cv2.imread("data/output/curated_masks/bigpatch_multimsk_Batch2 _9_c.tif", )
assert isinstance(bigsquare_multimsk, np.ndarray), "nope"

for msk in [multimsk, bigsquare_multimsk]:
    print(np.unique(msk.reshape(-1, msk.shape[2]), axis=0, return_inverse=True))
    layer_msk = stats.get_layer(msk, msk_col_dct, "petiole")
    plot_big(layer_msk)
    subs_metal_image = stats.get_sub_ele_img(img, layer_msk)
    plot_big(msk)
    print(stats.get_sub_ele_stats(subs_metal_image))
    
# %% Sensitivity analysis
randpix_df = pd.read_csv("data/rand_pred_pixel.csv", index_col=0, header=0)
para_df = pd.read_csv("data/sensitivity_paras.csv", index_col=0, header=0)
xy_tuplst = list(zip(randpix_df.x.tolist(), randpix_df.y.tolist()))
randpix_df["xy"] = xy_tuplst

with open(POLY_DCT_PATH, "rb") as f:
    polygon_dct = pickle.load(f)



for i in range(len(para_df)):
    blade_ksize, lap_ksize, thin_th, fat_th = para_df.loc[i, ['blade_ksize', 'lap_ksize', 'thin_th', 'fat_th']]
    randpix_df["pred_class_" +str(i)] =np.nan

    for batch in batchname_lst:
        multi_msk = processing.create_multimsks(batch, RAW_TIFF_PATH, BATCH_MSK_PATH, 
                             blade_ksize, lap_ksize, thin_th, fat_th,
                             msk_col_dct, BATCH_MULTIMSK_PATH)
        
        # Divide batch multimask into individual plants
        for acc_rep, polygon in polygon_dct[batch].items():
            accession, replicate = acc_rep.split("_")
            fn = "_".join([batch, accession, replicate])
            fn += ".tif"
            # Remove other plants
            bged_multimsk = processing.poly_crop(multi_msk, polygon, 
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
    
# %% Visualize confusion matrix
randpix_df = pd.read_csv("data/rand_pred_pixel.csv", index_col=0, header=0)

from sklearn import metrics
labels =  ["petiole", "margin", "vein", "tissue" ]
Y_obs = randpix_df["obs_class"]
Y_pred =  randpix_df["pred_class"]
conf_matrix = pd.DataFrame(metrics.confusion_matrix(Y_obs,Y_pred, [1,2,3,4]))
print(conf_matrix)
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cbar=False)
conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
plt.figure(figsize=(4,5))
p = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cbar=False)
p.set_xticklabels(labels, rotation=90)
p.set_yticklabels(labels, rotation=0)
plt.legend([],[], frameon=False)

# plt.savefig("data/output/article_images/confusion_matrix.png", dpi=300,
            # bbox_inches="tight")

plt.show()
F_scores = metrics.f1_score(Y_obs,Y_pred, labels = [1,2,3,4], average=None)

# F_scores = []
# for i in range(len(para_df)):
#     Y_pred =  randpix_df["pred_class_" + str(i)]
#     conf_matrix = pd.DataFrame(metrics.confusion_matrix(Y_obs,Y_pred, [1,2,3,4]))
#     conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
#     sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True)
#     F_scores.append(metrics.f1_score(Y_obs,Y_pred, labels = [1,2,3,4], average='weighted'))
#     plt.show()

# %% Visualize sensitivity analysis
from scripts.sensitivity_analysis import para_dct, para_map
from sklearn.metrics import confusion_matrix

sens_df = pd.read_csv("data/rand_pred_pixel_sens.csv", index_col=0, header=0)
para_df = pd.read_csv("data/sensitivity_paras.csv", index_col=0, header=0)

colnames = ["".join(("pred_class_",str(i))) for i in range(3,19)]
value_counts = sens_df[colnames].apply(pd.Series.value_counts)


sens_res_df = pd.DataFrame(columns=["Parameter","Parameter Value","Substructure","Accuracy"])
y_true = sens_df.obs_class
for k,v in para_map.items():
    colname = "".join(("pred_class_", str(k)))
    cm = confusion_matrix(y_true, sens_df[colname], labels = [1,2,3,4], normalize="true")
    acc = cm.diagonal()
    parameter = "_".join((v.split("_")[0], v.split("_")[1]))
    para_value = int(v.split("_")[2])
    for i, accuracy in enumerate(acc):
        substructure = class_dct[i + 1]
        sens_res_df = sens_res_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                          "Substructure":substructure,"Accuracy":accuracy},
                                         ignore_index=True)
        
# Repeat manually for blade_ksize parameter value 15 (default)
colname = "pred_class"
cm = confusion_matrix(y_true, sens_df[colname], labels = [1,2,3,4], normalize="true")
acc = cm.diagonal()
parameter = "blade_ksize"
para_value = 15
for i, accuracy in enumerate(acc):
    substructure = class_dct[i + 1]
    sens_res_df = sens_res_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                      "Substructure":substructure,"Accuracy":accuracy},
                                     ignore_index=True)
    
xlabs = {"blade_ksize":"blade opening kernel size [pixels]",
          "lap_ksize":"Laplacian kernel size [pixels]",
          "thin_th":"Threshold on Laplacian (thin) [-]",
          "fat_th": "Threshold on Laplacian (fat) [-]"}
fig, axs = plt.subplots(2,2,figsize=(10,10))
axs = axs.ravel()
for i, para in enumerate(sens_res_df.Parameter.unique().tolist()):
    # plt.subplot(2,2,i+1)
    sns.barplot(data=sens_res_df.loc[sens_res_df.Parameter == para,:],
        x="Parameter Value", y="Accuracy", hue="Substructure",
        palette=msk_hex_palette, ax=axs[i])
    axs[i].get_legend().remove()
    axs[i].set_xlabel(xlabs[para])
    axs[i].set_ylim(.4,1.0)
handles, labels = axs[-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower left', bbox_to_anchor= (0.0, 1.01))
# fig.legend(handles, labels, loc='lower left', bbox_to_anchor= (1, 1), ncol=2,
#             borderaxespad=0, frameon=False)
fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.8),loc = 'upper right')
plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.25)
# fig.savefig()
# plt.tight_layout()
# fig.legend(handles, labels)
# plt.tight_layout()
fig.savefig('data/output/article_images/sensitivity.png', bbox_inches='tight')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.1)

# %% Inspect erronous pixels
randpix_df = pd.read_csv("data/rand_pred_pixel.csv", index_col=0, header=0)
answer = ""
while answer != "stop":
    answer = input("class, stop: ")
    row = randpix_df.loc[(randpix_df["obs_class"] == int(answer)) & 
                         (randpix_df["obs_class"] != randpix_df["pred_class"]),:].sample()
    fn = row.fn.values[0]
    img = cv2.imread(PLANT_IMG_PATH + fn)
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn)

    print("true", row.obs_class, "pred", row.pred_class)
    x,y = row.x.values[0], row.y.values[0]
    img[x, y] = (0,0,255)
    cv2.circle(img, (y,x), 10, (0,0,255))
       
    nb = min(x + 50, img.shape[0])
    sb = max(0, x - 50)
    eb = max(0, y - 50)
    wb = min(y + 50, img.shape[1])
    plot_big2(img[sb:nb, eb:wb], multimsk[sb:nb, eb:wb])
    if input("save? ") == "yes":
        cv2.imwrite("data/output/article_images/wrongclass_img_" + answer + "_" + fn + ".png", img[sb:nb, eb:wb])
        cv2.imwrite("data/output/article_images/wrongclass_multimsk_" + answer + "_" + fn + ".png", multimsk[sb:nb, eb:wb])
        

    