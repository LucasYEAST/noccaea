# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:40:58 2020

@author: lucas
"""
from scripts import utils, segmentation, draw, stats, viz

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

msk_col_dct = viz.get_colors(obj_class_lst, "Set2")
msk_hex_palette = sns.color_palette(['#%02x%02x%02x' % (msk_col_dct[key][2], msk_col_dct[key][1], msk_col_dct[key][0]) \
                                     for key in obj_class_lst]) # BGR -> RGB -> HEX -> sns palette
msk_hex_palette = dict(zip(obj_class_lst, msk_hex_palette))


leaf_types = ["first", "grown_1", "grown_2", "developping"]
leafmsk_col_dct = viz.get_colors(leaf_types, "hls")


hex_msk_col_dct = {k:'#{:02x}{:02x}{:02x}'.format(v[2],v[1],v[0]) for k,v in msk_col_dct.items()}
RGB_df = pd.Series(hex_msk_col_dct)
RGB_df.to_csv("data/RGB_df.csv")

#TODO Write code to download data

# %% Manually divide batches into plants and annotate
df = pd.read_csv("data/Noccaea_nometrics.csv", index_col=0)

if os.path.exists(POLY_DCT_PATH):
    with open(POLY_DCT_PATH, "rb") as f:
        polygon_dct = pickle.load(f)
else:
    polygon_dct = {batchname:{} for batchname in batchname_lst}

for batch in batchname_lst:
    df = segmentation.divide_plants(RAW_TIFF_PATH, batch, polygon_dct, POLY_DCT_PATH, 
                                    df, DF_SAVE_PATH)

# %% Create batch foreground masks
if not os.path.exists(BATCH_MSK_PATH):
    os.mkdir(BATCH_MSK_PATH)

layers = ["Ca.tif", "K.tif", "Ni.tif",] # "Image.tif"
for batch in batchname_lst:
    segmentation.create_foreground_masks(RAW_TIFF_PATH, batch, layers, BATCH_MSK_PATH)
       
# %% Create batch segmentation masks

blade_ksize, lap_ksize, thin_th, fat_th = 15, 7, -2500, 0

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
    
    # Open batch images
    msk_path = BATCH_MSK_PATH + batch + "batchmsk.tif"
    msk = cv2.imread(msk_path,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(msk_path)
    
    img_path = RAW_TIFF_PATH + batch + "- Image.tif"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(img_path)
    
    multimsk_path = multimsk + batch + "multimsk.tif"
    multimsk = cv2.imread(multimsk_path) # Load as RGB
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(multimsk_path)
            
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


# %% Get stats from image
# TODO: profile time usage and improve speed

metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]
substructures = obj_class_lst[1:] + ["plant", "rand_1", "rand_2", "rand_5"]
df = pd.read_csv("data/Noccaea_nometrics.csv", index_col=0)

plant_fns = os.listdir(PLANT_MSK_PATH)
# noise loop
# for nlvl in ["10", "20", "50", "75", "90"]:
for metal in metals:
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
        
            # TODO noise levels loop (if still relevant)
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

# %% Find units for "absolute"
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
plt.title("plant size versus ICP:muXRF ratio per plant")
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
plt.title("IPC-AES versus muXRF per plant")
# plt.title("r: " + str(r))
plt.savefig("data/output/results/ICP-AES versus muXRF.png", bbox_inches='tight')
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
df = df.loc[df.batch.notna(),:]
random_accessions = random.sample(list(df["Accession #"].unique()), 15)
random_accessions.sort()
# acc_strs = [str(x) for x in random_accessions]

# Trying to draw error lines
plt_df = df.loc[df["Accession #"].isin(random_accessions),["Accession #", "metal_Z_plant_meanC"]]
line_x = plt_df["Accession #"].unique().tolist()
line_x = np.array(list(zip(line_x, line_x)))
line_y = np.zeros(line_x.shape)
for i,acc in enumerate(plt_df["Accession #"].unique().tolist()):
    line_y[i, 0] = plt_df.loc[plt_df["Accession #"] == acc, "metal_Z_plant_meanC"].min()
    line_y[i, 1] = plt_df.loc[plt_df["Accession #"] == acc, "metal_Z_plant_meanC"].max()
                  
plt.figure(figsize=(7,5))
sns.catplot(x="Accession #", y="metal_Z_plant_meanC", data=plt_df, jitter=False, color='black') #hue="Accession #", palette=msk_hex_palette, legend=False
# plt.plot(line_x, line_y)
plt.ylabel("mean zinc concentration [-]")
plt.savefig("data/output/article_images/mean_zinc_concentration.png", dpi=300)

# %% Histograms of metals
sns.reset_orig()
metals = ["Zn","Ca","K","Ni"]
minlst = []
maxlst = []
meanlst = []
stdlst = []

for metal in metals:
    print(metal)
    for batch in batchname_lst:
        metalimg_path = RAW_TXT_PATH + batch + "- " + metal + ".txt"
        img = np.loadtxt(metalimg_path, delimiter=",", skiprows=1).flatten()
        print(img.mean(), img.std(), img.min(), img.max())
        plt.hist(img.ravel(), bins=256, range=(img.min(), img.max()))
        plt.title(metal + " " + batch)
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


# %% scatter substructure CQs against each other within metal
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

# %% Plot H2 for all metals
# TODO create grouped barchart (add columns for metal and substructure in R script)

H2 = pd.read_csv("data/H2_table.csv", index_col=0)
# sns.set_style("ticks")
substructures = obj_class_lst[1:]
metals = ['Z', 'K', 'Ni', 'Ca']
metrics = ["plant_n_pix", "plant_meanC"] + substructures + \
    ["rand_1"] #, "rand_10"]
i_names = ["plant size", "mean [Zn]", "petiole CQ", "margin CQ", "vein CQ", "tissue CQ", 
           "random \nsubstructure \nCQ"] #, "random 10 CQ"]
colors = [np.array((105,105,105))/256, np.array((169,169,169))/256] + \
    [msk_hex_palette[s] for s in substructures] + \
        [np.array((119,136,153)) / 256] #, np.array((119,136,153))/256]
palette = {k:v for k,v in zip(i_names, colors)}


bars = ["_".join((m, "metal", metals[0])) for m in metrics]
data = H2.loc[bars, "H2_percent"]
data.index = i_names
sns.barplot(x=data.index, y=data.values, palette=colors)
plt.ylabel("H2 (%)")
plt.xticks(rotation=45)
plt.ylim((0,100))
plt.savefig("data/output/article_images/CQSizeMeanRand_H2.png", dpi=300, bbox_inches="tight")
plt.show()

xlabs = ["Zn", "K", "Ni", "Ca"]
# plt.figure(figsize=(10,10))
indexnames = []
sub_names = []

for i, metal in enumerate(metals):
    indexnames += ["_".join((substrct, "metal", metal)) for substrct in substructures]
    sub_names += substructures
metal_names = [metals[0]]*4 + [metals[1]]*4 + [metals[2]]*4 + [metals[3]]*4
    
data = pd.DataFrame(H2.loc[indexnames,"H2_percent"])
data['substructures'] = sub_names
data['metals'] = metal_names
# plt.subplot(2,2,i + 1)
sns.barplot(x=data.metals, y=data.H2_percent, 
            hue=data.substructures, palette=msk_hex_palette) #(data=data,x=data.index, y=data.H2_percent, hue=data.index)
# plt.title(plt_titles[i])
plt.ylabel("H2 (%)")
plt.xlabel("")
plt.xticks(ticks=range(4), labels=xlabs, rotation=0)
plt.ylim((0,100))
plt.legend(bbox_to_anchor=(1.32, 1),loc = 'upper right')
plt.savefig("data/output/article_images/CQ_H2_substrs.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot H2 for noised images
H2_noised = pd.read_csv("data/H2_noised_table.csv", index_col=0)
indexnames = []
sub_names = []
metal = "Z"
noise_levels = ["10", "20", "50", "75", "90"]
noise_names = [10,20,50,75,90] * 4
noise_names.sort()
noise_names = list(map(str, noise_names))

for nlvl in noise_levels:
    indexnames += ["_".join(("noise", nlvl, substrct, "metal", metal)) for substrct in substructures]
    sub_names += substructures


data = pd.DataFrame(H2_noised.loc[indexnames, "H2_percent"])
data['substructures'] = sub_names
data['noise level'] = noise_names
sns.barplot(x=data["noise level"], y=data.H2_percent, 
            hue=data.substructures, palette=msk_hex_palette) #(data=data,x=data.index, y=data.H2_percent, hue=data.index)
plt.ylabel("H2 (%)")
plt.xlabel("Pixels changed to random class (%)")
plt.xticks(ticks=range(5), labels=noise_levels, rotation=0)
plt.ylim((0,100))
plt.legend(bbox_to_anchor=(1.32, 1),loc = 'upper right')
plt.savefig("data/output/article_images/CQ_H2_Noise_substrs.png", dpi=300, bbox_inches="tight")


# %% Get correlations between metals within substructures
df = pd.read_csv("data/Noccaea_CQsA500.csv")
df_noNAN = df.loc[df['batch'].notna(),:]

def return_pval(x,y):
    return pearsonr(x, y)[1]

substructures = obj_class_lst[1:]
metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]

for substr in substructures:
    pairplt_vars = [ "_".join((metal, substr, "CQ")) for metal in metals]
    CQmetal_pair_corrs = df_noNAN[pairplt_vars].corr(method='pearson')
    CQmetal_pair_corrs.columns = metals
    CQmetal_pair_corrs.index = metals
    sns.heatmap(CQmetal_pair_corrs, annot=True, cbar=False, cmap="viridis")
    plt.title(substr)
# plt.savefig("data/output/article_images/CQ_paircorr.png", dpi=300)

    plt.show()    
    CQ_pair_cor_pval = df_noNAN[pairplt_vars].corr(method=return_pval)
    g = sns.pairplot(df_noNAN[pairplt_vars]) # , 
    plt.show()

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
    
# %% Visualize sensitivity analysis
# TODO fix the manual blade size part in tandem with sensitivity analysis itself
from scripts.sensitivity_analysis import para_dct, para_map
from sklearn.metrics import confusion_matrix, f1_score


sens_df = pd.read_csv("data/rand_pred_pixel_sens.csv", index_col=0, header=0)
para_df = pd.read_csv("data/sensitivity_paras.csv", index_col=0, header=0)

colnames = ["".join(("pred_class_",str(i))) for i in range(3,19)]
value_counts = sens_df[colnames].apply(pd.Series.value_counts)


sens_res_df = pd.DataFrame(columns=["Parameter","Parameter Value","Substructure","Accuracy"])
F1_df = pd.DataFrame(columns=["Parameter","Parameter Value","Substructure","F1"])
y_true = sens_df.obs_class
F_scores = {}
for k,v in para_map.items():
    colname = "".join(("pred_class_", str(k)))
    cm = confusion_matrix(y_true, sens_df[colname], labels = [1,2,3,4], normalize="true")
    acc = cm.diagonal()
    parameter = "_".join((v.split("_")[0], v.split("_")[1]))
    para_value = int(v.split("_")[2])
    F1_scores = f1_score(y_true,sens_df[colname], labels = [1,2,3,4], average=None)

    for i, accuracy in enumerate(acc):
        substructure = class_dct[i + 1]
        sens_res_df = sens_res_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                          "Substructure":substructure,"Accuracy":accuracy},
                                         ignore_index=True)
        F1_df = F1_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                          "Substructure":substructure,"F1":F1_scores[i]},
                                         ignore_index=True)
        
# Repeat manually for blade_ksize parameter value 15 (default)
colname = "pred_class"
cm = confusion_matrix(y_true, sens_df[colname], labels = [1,2,3,4], normalize="true")
acc = cm.diagonal()
parameter = "blade_ksize"
para_value = 15

F1_scores = f1_score(y_true,sens_df[colname], labels = [1,2,3,4], average=None)

for i, accuracy in enumerate(acc):
    substructure = class_dct[i + 1]
    sens_res_df = sens_res_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                      "Substructure":substructure,"Accuracy":accuracy},
                                     ignore_index=True)
    F1_df = F1_df.append({"Parameter":parameter,"Parameter Value":para_value,
                                          "Substructure":substructure,"F1":F1_scores[i]},
                                         ignore_index=True)
    
xlabs = {"blade_ksize":"blade opening kernel size [pixels]",
          "lap_ksize":"Laplacian kernel size [pixels]",
          "thin_th":"Threshold on Laplacian (thin) [-]",
          "fat_th": "Threshold on Laplacian (fat) [-]"}

# Plot accuracy
fig, axs = plt.subplots(2,2,figsize=(10,10))
axs = axs.ravel()
for i, para in enumerate(sens_res_df.Parameter.unique().tolist()):
    # plt.subplot(2,2,i+1)
    sns.barplot(data=sens_res_df.loc[sens_res_df.Parameter == para,:],
        x="Parameter Value", y="Accuracy", hue="Substructure",
        palette=msk_hex_palette, ax=axs[i])
    axs[i].get_legend().remove()
    axs[i].set_xlabel(xlabs[para])
    axs[i].set_ylim(0.,1.0)
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.8),loc = 'upper right')
plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.25)
fig.savefig('data/output/article_images/Acc_sensitivity.png', bbox_inches='tight')

# Plot F1 scores
fig, axs = plt.subplots(2,2,figsize=(10,10))
axs = axs.ravel()
for i, para in enumerate(F1_df.Parameter.unique().tolist()):
    # plt.subplot(2,2,i+1)
    sns.barplot(data=F1_df.loc[F1_df.Parameter == para,:],
        x="Parameter Value", y="F1", hue="Substructure",
        palette=msk_hex_palette, ax=axs[i])
    axs[i].get_legend().remove()
    axs[i].set_xlabel(xlabs[para])
    axs[i].set_ylim(0.,1.0)
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.8),loc = 'upper right')
plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.25)
fig.savefig('data/output/article_images/F1_sensitivity.png', bbox_inches='tight')

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
    viz.plot_big2(img[sb:nb, eb:wb], multimsk[sb:nb, eb:wb])
    if input("save? ") == "yes":
        cv2.imwrite("data/output/article_images/wrongclass_img_" + answer + "_" + fn + ".png", img[sb:nb, eb:wb])
        cv2.imwrite("data/output/article_images/wrongclass_multimsk_" + answer + "_" + fn + ".png", multimsk[sb:nb, eb:wb])
        
    
# %% Create noised masks
for percentage in [90]: 
    segmentation.create_noised_msks(PLANT_MULTIMSK_PATH, PLANT_MSK_PATH, plant_fns, msk_col_dct, percentage)
# %% Create random substructures
for N_pixels in [1,2,5]:
    segmentation.create_rand_substructure(PLANT_MSK_PATH, PLANT_RANDMSK_PATH, N_pixels)
    
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
    contours = segmentation.contouring(blade_msk.astype("uint8"))
    accepted_contours = []
    for cnt in contours:
        cnt_img = cv2.drawContours(img.copy(), [cnt], 0, (0,255,0), 1)
        viz.plot_big(cnt_img)
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
                bl_msk_cnt = segmentation.contouring(manual_blade_msk.astype("uint8"))[0]
                new_cnt_img = cv2.drawContours(img.copy(), [bl_msk_cnt], 0, (0,255,0), 1)
                viz.plot_big(new_cnt_img)
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
# TODO write as function, move to segmentation.py

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
