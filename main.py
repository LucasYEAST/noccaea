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
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(69)

RAW_TIFF_PATH = "data/raw_data/tiff/"
RAW_TXT_PATH = "data/raw_data/txt/"

BATCH_MSK_PATH = "data/batch_msk/"
BATCH_MULTIMSK_PATH = "data/batch_multimsk/"

PLANT_IMG_PATH = "data/plant_img/"
PLANT_MSK_PATH = "data/plant_msk/"
PLANT_MULTIMSK_PATH = "data/plant_multimsk/"
PLANT_RANDMSK_PATH = "data/plant_randmsk/"

PLANT_ZIMG_PATH = "data/plant_Zimg/"
PLANT_ZIMG_NOISE_PATH = "data/plant_Zimg_noise/"
PLANT_KIMG_PATH = "data/plant_Kimg/"

DF_SAVE_PATH = "data/Noccaea_processed.csv"
POLY_DCT_PATH = "data/polygon_dict.pck"

batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)

obj_class_lst = ["background", "petiole", "margin", "vein", "tissue" ]
msk_col_dct = get_colors(obj_class_lst)


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
# %%
## Get vein mask
blade_img = np.where(blade, img, 0)
lap_img = cv2.Laplacian(blade_img,cv2.CV_64F, ksize=7)

lap_lower0 = np.where(lap_img < 0, 255, 0)
vein_mask = np.where((lap_lower0 == 255) & (blade == 255), 255, 0)

lap_lower_minus1000 = np.where(lap_img < -1000, 255, 0)
vein_mask_th = np.where((lap_lower_minus1000 == 255) & (blade == 255), 255, 0)

lap_lower_minus2500 = np.where(lap_img < -2500, 255, 0)
vein_mask_lower2500 = np.where((lap_lower_minus2500 == 255) & (blade == 255), 255, 0)
skeleton_veins = cv2.ximgproc.thinning(vein_mask.astype("uint8"))
veins = cv2.add(skeleton_veins,vein_mask_lower2500.astype("uint8") ) 


marginless_veins = np.where((vein_mask_th == 255) & (margin == 0), 255, 0)

blade_img = cv2.GaussianBlur(blade_img,(5,5), sigmaX=0)
th_veins = cv2.adaptiveThreshold(blade_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,7,0)
th_veins = np.where((blade == 255) & (margin == 0), th_veins, 0)

noise_kernel = np.ones((2,2),np.uint8)
th_veins_opened = cv2.morphologyEx(th_veins, cv2.MORPH_OPEN, noise_kernel)


plot_big(veins[:500:,:500], title="added skeleton and lap lower -2500")
cv2.imwrite("data/output/added_laplace_skeleton_gray.tif", veins[:300,:300])


plt.figure(figsize=(25,25))
plt.subplot(2,2,1),plt.imshow(img[:300,:300], 'gray')
plt.title("Compton scatter image", fontsize=30)
plt.subplot(2,2,2),plt.imshow(vein_mask[:300,:300],'gray')
plt.title("Laplace <0", fontsize=30)
plt.subplot(2,2,3),plt.imshow(skeleton_veins[:300,:300], 'gray')
plt.title("Skeletonized adaptive threshold", fontsize=30)
plt.subplot(2,2,4),plt.imshow(vein_mask_lower2000[:300,:300], 'gray')
plt.title("Laplacian <-2500 negative", fontsize=30)
plt.savefig("data/output/vein_comparison2.png")
plt.show()


# Convert skeletonized to red and laplacian -2500 to green
img_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_BGR[:,:,1] = 0
skeleton_veins = cv2.subtract(skeleton_veins, vein_mask_lower2000.astype("uint8"))
skeleton_veins_BGR = cv2.cvtColor(skeleton_veins, cv2.COLOR_GRAY2BGR)
# skeleton_veins_BGR[:,:,1] = 0

vein_mask_lower2000_BGR = cv2.cvtColor(vein_mask_lower2000.astype("uint8"), cv2.COLOR_GRAY2BGR)
# vein_mask_lower2000_BGR[:,:,0] = 0

addition = cv2.add(vein_mask_lower2000_BGR, skeleton_veins_BGR)
add_ov = overlay(addition,img_BGR, alpha=0.5, beta=0.5)
sub = cv2.subtract(img_BGR, addition)


cv2.imwrite("data/output/added_laplace_skeleton.tif", add_ov[:300,:300])
# plot_big(addition)


# num_labels, labels_im = cv2.connectedComponents(th_veins_opened, connectivity=4)

# def imshow_components(labels):
#     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

#     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

#     # set bg label to black
#     labeled_img[label_hue==0] = 0

#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()

# imshow_components(labels_im)

# skeleton_veins_terts = cv2.ximgproc.thinning(vein_mask.astype("uint8"))
# plt.figure(figsize=(10,10))
# plt.title("Skeletonized Laplacian <0")
# plt.imshow(skeleton_veins_terts[:300,:300], 'gray')
# plt.savefig("data/output/vein_comparison_LapSkel.png")


# %% Create individual images per plant
metals = ["Ni", "Ca"]

for metal in metals:
    processing.make_individual_plant_images(POLY_DCT_PATH, batchname_lst, RAW_TIFF_PATH, 
                                 BATCH_MSK_PATH, BATCH_MULTIMSK_PATH, RAW_TXT_PATH, 
                                 PLANT_IMG_PATH, PLANT_MSK_PATH, PLANT_MULTIMSK_PATH,
                                 metal, msk_col_dct, create_masks = False)

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

# %% Review results for random insertion of class noise
df = pd.read_csv("data/Noccaea_proc_Znoise.csv")
sampled_accessions = list(set(np.random.choice(df['Accession #'], 25)))
sample = df.loc[df['Accession #'].isin(sampled_accessions),:]

plt.scatter(sample['Accession #'], sample["vein_CQ"], c="blue", label="CQ vein", s=3)
# plt.scatter(sample['Accession #'], sample["vein_noise10_CQ"], c="orange", 
#             label="CQ vein 10% rand intensity", s=3)
plt.scatter(sample['Accession #'], sample["vein_noise20_CQ"], c="red", 
            label="CQ vein 20% rand intensity", s=3)

plt.ylabel("Concentration Quotient")
plt.xlabel("accession label")
plt.legend()
plt.ylim(0.9,1.2)
# plt.subplot((212))
plt.savefig("data/output/plots/CQ normal vs random vein CQ.png")
plt.show()


plt.scatter(sample['Accession #'], sample["vein_meanZC"], c="blue", label="meanZC margin")
# plt.scatter(sample['Accession #'], sample["vein_noise10_meanZC"], c="orange", label="meanZC vein 10% rand intensity")
plt.scatter(sample['Accession #'], sample["vein_noise20_meanZC"], c="red", label="meanZC vein 20% rand intensity")

plt.ylabel("mean pixel intensity")
plt.xlabel("accession label")
plt.legend()
plt.savefig("data/output/plots/mean pixel intensity normal versus random vein.png")
plt.show()

# %% Scatter CQs against n_pixel and mean plant CQ
df = pd.read_csv("data/Noccaea_proc_Znoise100.csv")
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
df = pd.read_csv("data/Noccaea_proc_Znoise.csv")
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


# %% Absolute metal of 500 pixels
df = pd.read_csv("data/Noccaea_proc_ZK.csv", index_col=0)
substructure_lst = obj_class_lst[1:] # All objects except background + whole plant
substructure_lst += ["plant"]
substructure_lst = [subs + "_K_abs500pix" for subs in substructure_lst]

METAL_PATH = PLANT_KIMG_PATH
for fn in os.listdir(PLANT_MSK_PATH):
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
    
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
    
    metal_fn = fn.split(".")[0] + ".txt"
    img = np.genfromtxt(METAL_PATH + metal_fn, delimiter=",")
    
    abs500pix_lst = []
    for subs in substructure_lst:
       if "plant" in subs:
           layer_msk = msk * 255
       else:
           layer_msk = stats.get_layer(multimsk, msk_col_dct, subs.split("_")[0])
       abs500pix_lst.append(stats.XrandPixel_value(layer_msk, img, metal_fn, subs, 500))

    df.loc[df["fn"] == fn, substructure_lst] = abs500pix_lst

df.to_csv("data/Noccaea_proc_ZK.csv")
# %% check sizes of substructures

    

# %% Create df with random CQs to test
phenotypes = ("plant_npixel","petiole_CQ","margin_CQ","vein_CQ","tissue_CQ")
genotype_n = 86
genotype = np.repeat(range(genotype_n), 3)
replicate = ["a","b","c"] * genotype_n

random_df =  pd.DataFrame({"Accession..":genotype, "Biological.replicate":replicate})
for pt in phenotypes:
    random_df[pt] = np.random.random(len(random_df)) * 2
    
random_df.to_csv("data/random_CQs.csv")

# %% Create noised Zimages

# Per image create histogram of pixel intensity under mask and pick random values from 
# histogram for x% of pixels
for perc in [100]: #10, 20
    for fn in os.listdir(PLANT_MSK_PATH):
        # plant_msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
        # assert isinstance(plant_msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
        
        Zfn = fn.split(".")[0] + ".txt"
        Zimg = np.genfromtxt(PLANT_ZIMG_PATH + Zfn, delimiter=",")
        Zimg_1D = Zimg.ravel()
        hist = np.delete(Zimg_1D, np.argwhere(Zimg_1D == 0)) # Get all non-background values
        N = int(Zimg.size * (perc / 100))
        random_pixels = (np.random.randint(0, Zimg.shape[0] - 1, size=N), 
                          np.random.randint(0, Zimg.shape[1] - 1, size=N))
        random_values = np.random.choice(hist, size=N)
        Zimg[random_pixels] = random_values
        np.savetxt(PLANT_ZIMG_NOISE_PATH + str(perc) + "/" + Zfn, Zimg, fmt='%f', delimiter=",")
        
# %% Create random substructure
import random

for N_pixels in [5]:
    for fn in os.listdir(PLANT_MSK_PATH):
        
        # Load plant mask
        msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
        assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
        
        # Create empty mask
        rand_msk = np.zeros(msk.shape, dtype="uint8")
        
        # Select 10 random pixels
        x,y = np.where(msk == 1 )
        i_lst = random.sample(range(len(x)), N_pixels)
        rand_msk[x[i_lst], y[i_lst]] = 1
        
        # Dilate pixels (multiple times?)
        kernel = np.ones((15,15),np.uint8)
        dil_msk = cv2.dilate(rand_msk, kernel, iterations = 1)
        
        # Crop to plant mask
        dil_msk = dil_msk & msk
        dil_msk = dil_msk * 255
        
        cv2.imwrite(PLANT_RANDMSK_PATH + str(N_pixels) + "/" + fn,  dil_msk)

# %% Calculate CQ for random substructures 
df = pd.read_csv("data/Noccaea_proc_ZK.csv", index_col=0)
N_rand_pix_lst = [10, 20, 30, 40]
phenotypes = ["rand_Z_CQ_" + str(npix) for npix in N_rand_pix_lst]
subs_abs_colnames = ["rand_Z_abs_" + str(npix) for npix in N_rand_pix_lst]
subs_n_colnames = ["rand_npixel_" + str(npix) for npix in N_rand_pix_lst]
subs_mean_colnames = ["rand_Z_meanC_" + str(npix) for npix in N_rand_pix_lst]


METAL_PATH = PLANT_ZIMG_PATH

for fn in os.listdir(PLANT_MSK_PATH):
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(PLANT_MSK_PATH + fn)
    
    metal_fn = fn.split(".")[0] + ".txt"
    img = np.genfromtxt(METAL_PATH + metal_fn, delimiter=",")
    
    abslst, npixlst, meanClst = [], [], []
    for N_pixels in N_rand_pix_lst:
        rand_msk = cv2.imread(PLANT_RANDMSK_PATH + str(N_pixels) + "/" + fn,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
        assert isinstance(rand_msk, np.ndarray), "{} doesn't exsit".format(PLANT_RANDMSK_PATH + fn)
    
        subs_metal_image = stats.get_sub_ele_img(img, rand_msk)
        abs_metal, n_pixels, meanC = stats.get_sub_ele_stats(subs_metal_image)
        abslst.append(abs_metal)
        npixlst.append(n_pixels)
        meanClst.append(meanC)
        
    df.loc[df["fn"] == fn, subs_abs_colnames] = abslst
    df.loc[df["fn"] == fn, subs_n_colnames] = npixlst
    df.loc[df["fn"] == fn, subs_mean_colnames] = meanClst

df[phenotypes] = df[subs_mean_colnames].div(df["plant_meanZC"], axis=0)
df.to_csv("data/Noccaea_proc_ZK.csv")
    
# %% Metal correlations
metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]

metal_cor_df = pd.DataFrame(columns = metals)
Z, K, Ni, Ca = [], [], [], []

plant_fns = os.listdir(PLANT_MSK_PATH)
# plant loop
for fn in plant_fns:       
    # metals loop
    for metal in metals:
        


    