# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:40:36 2020

@author: lucas
"""
from scripts import utils, stats, viz

import cv2
import numpy as np
import pandas as pd

import os
import random
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.offsetbox import AnchoredText

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

if not os.path.exists("data/output/article_images/"):
    os.makedirs("data/output/article_images/")
    
# %% Output batch scan examples
metals = ["Zn", "K", "Ca", "Ni"]
fn = batchname_lst[0]
compton_fn = fn + "- " + "Image" + ".tif"

# plt.subplot(2,2,)
compton = cv2.imread(RAW_TIFF_PATH + compton_fn, cv2.IMREAD_GRAYSCALE)
plt.imshow(compton[:600,:600], cmap="gray")
plt.axis("off")
plt.colorbar()
plt.savefig("data/output/article_images/compton_scatter.png", dpi=300)

plt.figure(figsize=(10,10))
for i, metal in enumerate(metals):
    plt.subplot(2,2,i+1)
    metal_fn = fn + "- " + metal + ".txt"
    img = np.genfromtxt(RAW_TXT_PATH + metal_fn, delimiter=",", skip_header=1)
    plt.imshow(img[:600,:600], cmap='gray')
    plt.axis("off")
    plt.title(metal)
    plt.colorbar()

plt.savefig("data/output/article_images/metal_images.png", dpi=300)
    # viz.plot_big(img[:600,:600])

# %% Correlate ICP and muXRF
df = pd.read_csv("data/Noccaea_CQs.csv")
sns.set_style("ticks")
palette = itertools.cycle(sns.color_palette())

metals = ["Zn","Ca","K","Ni"]

# Correlate ICP-AES ratio to plant size for all metals
for metal in metals:
    if metal == "Zn":
        abs_metal = "metal_Z_plant_abs"
    else:
        abs_metal = "_".join(("metal", metal, "plant", "abs"))
    
    ratio = df[metal] / df[abs_metal]
    sns.scatterplot(x=df["metal_Z_plant_n_pix"], y = ratio, color=next(palette))
    plt.ylabel("ICP-AES : \u03BCXRF ratio")
    plt.xlabel("Plant Size [pixels]")
    plt.savefig("data/output/article_images/ICP-AES_muXRF_ratio_plantsize"+ metal + ".png", bbox_inches="tight")
    plt.show()

# Correlate mean zinc concentration to the ICP:muXRF ratio
# sns.scatterplot(x="metal_Z_plant_meanC", y = "ICP:muXRF_ratio", data=df)
# plt.ylabel("ICP-AES : muXRF ratio")
# plt.xlabel("mean Zinc concentration")
# plt.show()

# Correlate ICP-AES zinc concentration to muXRF zinc concentration
# sns.scatterplot(x="Zn", y = "metal_Z_plant_abs", data=df)
# dfcor = df.loc[(df.Zn.notna()) & (df.metal_Z_plant_abs.notna()),:]
# r,p = pearsonr(dfcor.Zn, dfcor.metal_Z_plant_abs)
# print(r,p)
# plt.ylabel("\u03BCXRF absolute zinc [-]")
# plt.xlabel("ICP-AES absolute zinc [\u03BCm]")
# plt.title("IPC-AES versus muXRF per plant")
# # plt.title("r: " + str(r))
# plt.savefig("data/output/results/ICP-AES versus muXRF.png", bbox_inches='tight')
# plt.show()

# Find outliers
# outlier_abs_fn = df.loc[df["ICP:muXRF_ratio"] > 0.00075,"fn"].tolist()
# plt.scatter(df.index, df["ICP:muXRF_ratio"])
# plt.title("measured Zinc : machine vision zinc ratio")
# plt.show()

# Show images for outliers
# metal_name = "Z"
# METAL_PATH = "data/plant_" + metal_name + "img/"

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
df = pd.read_csv("data/Noccaea_CQs.csv", index_col=0)
df = df.loc[df.batch.notna(),:]
random_accessions = random.sample(list(df["Accession #"].unique()), 15)
random_accessions.sort()
plt_df = df.loc[df["Accession #"].isin(random_accessions),["Accession #", "metal_Z_plant_meanC"]]


plt.figure(figsize=(7,5))
sns.catplot(x="Accession #", y="metal_Z_plant_meanC", data=plt_df, jitter=False, color='black')
plt.ylabel("mean zinc concentration [-]")
# plt.savefig("data/output/article_images/mean_zinc_concentration.png", dpi=300)

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
        plt.show()
    
# %% Correlate metals

metals = ["Z","Ca","K","Ni"]
met_tup = itertools.combinations(metals, 2)
met_str = [" vs ".join(tup) for tup in met_tup]
metal_cor_dct = {"plant_fn":[], "accession #":[], "pairwise correlation":[], "r":[], "p":[]}

for fn in plant_fns:
    acc_nr = fn.split("_")[1]
    
    znimg = np.genfromtxt( "data/plant_Zimg/" + fn.split(".")[0] + ".txt", 
                          delimiter=",")
    kimg = np.genfromtxt( "data/plant_Kimg/" + fn.split(".")[0] + ".txt", 
                          delimiter=",")
    niimg = np.genfromtxt( "data/plant_Niimg/" + fn.split(".")[0] + ".txt", 
                          delimiter=",")
    caimg = np.genfromtxt( "data/plant_Caimg/" + fn.split(".")[0] + ".txt", 
                          delimiter=",")
    
    # Get pixel-values under plant mask
    msk = cv2.imread(PLANT_MSK_PATH + fn,  cv2.IMREAD_GRAYSCALE) // 255
    pixval_dct = {"Z":znimg[msk == 1], "Ca":caimg[msk == 1], 
                  "K":kimg[msk == 1], "Ni":niimg[msk == 1]}    

    # Calculate correlations        
    for string in met_str:
        metal_cor_dct["accession #"].append(acc_nr)
        metal_cor_dct["plant_fn"].append(fn)
        metal_cor_dct["pairwise correlation"].append(string)
        m1, m2 = string.split(" vs ")
        r,p = pearsonr(pixval_dct[m1],pixval_dct[m2])
        metal_cor_dct["r"].append(r)
        metal_cor_dct["p"].append(p)

# %% Show metal correlations
metal_cor_df = pd.DataFrame(metal_cor_dct)
metal_cor_df["accession #"] = metal_cor_df["accession #"].astype(int)
metal_cor_df.sort_values(by="accession #", axis=0, inplace=True)

sns.stripplot(x="pairwise correlation", y="r", data=metal_cor_df, linewidth=1)
plt.legend("")
plt.ylabel("$\it{r}$")
plt.xlabel("")
plt.ylim(-1,1)
plt.savefig("data/output/article_images/metal_cor.png", dpi=300, bbox_inches="tight")
plt.show()

accs = random.sample(metal_cor_df["accession #"].unique().tolist(), 15)
plot_df = metal_cor_df.loc[metal_cor_df["accession #"].isin(accs),:]
sns.catplot(x="accession #", y="r", data=plot_df, hue="pairwise correlation", jitter=False, linewidth=.5, s=5)
plt.legend("")
plt.ylim(-1,1)
# fig = plt.figure(figsize=(10,10))
plt.show()
for i, pair in enumerate(met_str):
    # ax = fig.add_subplot(2,3,i+1)
    print(pair)
    plt.figure(figsize=(20,8))
    sns.stripplot(x="accession #", y="r", 
                data=metal_cor_df.loc[metal_cor_df["pairwise correlation"] == pair,:], 
                jitter=False, palette="Set2")
    plt.xticks(fontsize=10, rotation=45)
    plt.ylim(-1,1)
    plt.savefig("data/output/article_images/sup_metalcor_"+pair+".png", dpi=300)
    plt.show()

# %% Get normalized mean concentration of substructures
sns.set_style("ticks")
mean_subC_dct = {substrct:[] for substrct in obj_class_lst[1:]}
mean_subC_dct.update({"metal":[], "plant_fn":[]})
metals = ["Z","Ca","K","Ni"]
for fn in plant_fns:
    multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn)
    msk = cv2.imread(PLANT_MSK_PATH + fn, cv2.IMREAD_GRAYSCALE) // 255
    for metal in metals:
        mean_subC_dct["metal"].append(metal) # annotate metal for dataframe
        mean_subC_dct["plant_fn"].append(fn) # annotate plant file name for dataframe
        # Open metal images
        METAL_PATH = "data/plant_" + metal + "img/"
        img = np.genfromtxt(METAL_PATH + fn.split(".")[0] + ".txt", delimiter=",")
        
        # Z-normalize plant concentrations with plant mean and std
        img_mean = img[msk > 0].mean()
        img_std = img[msk > 0].std()
        img_norm = (img - img_mean) / img_std # Z-normalization
        
        # Calculate mean concentration of substructures
        for substrct in obj_class_lst[1:]:
            layer_msk = stats.get_layer(multimsk, msk_col_dct, substrct)
            subs_metal_image = stats.get_sub_ele_img(img_norm, layer_msk)
            meanC = subs_metal_image[layer_msk > 0].mean()
            mean_subC_dct[substrct].append(meanC)
        
# Create dataframe
mean_subC_df = pd.DataFrame(mean_subC_dct)
mean_subC_df.to_csv("data/normalized_mean_subs_conc.csv")


# %% Check normalized mean concentrations for significant differences
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, kruskal, probplot
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols

metals = ["Z","Ca","K","Ni"]
mean_subC_df = pd.read_csv("data/normalized_mean_subs_conc.csv", index_col=0, header=0)

# Plot mean normalized concentrations
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
axes = axes.ravel()
for i, metal in enumerate(metals):
    df = mean_subC_df.loc[mean_subC_df["metal"] == metal,:]
    sns.boxplot(data=df, palette=msk_hex_palette, ax=axes[i])
    anchored_text = AnchoredText(metal, loc=1)
    axes[i].add_artist(anchored_text)
    axes[i].set_ylim(-2, 2.5)
    
    if i == 0:
        axes[i].set_ylabel("plant mean normalized concentration")
        
plt.savefig("data/output/article_images/normalized_mean_substructure_conc.png", dpi=300,
            bbox_inches="tight")
plt.show()

## Find significant differences in mean normalized concentrations
## Checking assumptions of normalcy: REJECTED

# Show distributions
# for metal in metals:
#     df = mean_subC_df[mean_subC_df["metal"] == metal]
#     for substr in obj_class_lst[1:]:
#         # df[substr].hist(bins=len(df)//2, color=msk_hex_palette[substr])
#         stat, p = shapiro(df[substr])
#         print('Statistics=%.3f, p=%.3f' % (stat, p))
#         # interpret
#         alpha = 0.05
#         if p > alpha:
#          	print('{}, sub: {}: Sample looks Gaussian (fail to reject H0)'.format(metal, substr))
#         else:
#          	print('{}, sub: {}: Sample does not look Gaussian (reject H0)'.format(metal, substr))
#     plt.hist(df["petiole"], bins=len(df)//4,alpha=0.5, color=msk_hex_palette["petiole"])
#     plt.hist(df["margin"], bins=len(df)//4,alpha=0.5, color=msk_hex_palette["margin"])
#     plt.hist(df["vein"],bins=len(df)//4,alpha=0.5, color=msk_hex_palette["vein"])
#     plt.hist(df["tissue"], bins=len(df)//4, alpha=0.5,color=msk_hex_palette["tissue"])
#     plt.title(metal)
#     plt.show()

# Test ANOVA for assumptions
# for metal in metals:
#     df = mean_subC_df[mean_subC_df["metal"] == metal].drop(["metal", "plant_fn"], axis="columns")
#     df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
#     df_melt.columns = ['index', 'treatments', 'value']
#     df_melt = df_melt.drop("index", axis="columns")
#     # Ordinary Least Squares (OLS) model
#     model = ols('value ~ C(treatments)', data=df_melt).fit()
#     stat, p = shapiro(model.resid)
#     alpha = 0.05
#     if p > alpha:
#         print(metal + ': Residuals look Gaussian (fail to reject H0)')
#     else:
#         print(metal + ': Residuals do not look Gaussian (reject H0)')
#     normality_plot, stat = probplot(model.resid, plot= plt, rvalue= True)
#     plt.show()
    # anova_table = sm.stats.anova_lm(model, typeric tests)

# Use non-parametric test to find significant differences
for metal in metals:
    df = mean_subC_df[mean_subC_df["metal"] == metal]
    print(metal, ": ", len(df))
    stat, p = kruskal(df["petiole"],df['margin'],df['vein'],df['tissue'],nan_policy ="omit")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    # interpret
    alpha = 0.05
    if p > alpha:
     	print('Same distributions (fail to reject H0)')
    else:
     	print('Different distributions (reject H0)')
    
    post = sp.posthoc_dunn([df["petiole"],df['margin'],df['vein'],df['tissue']], p_adjust = 'bonferroni')
    print(post)

# %% Scatter absolute metal concentration against plant size and mean zinc concentration
df = pd.read_csv("data/Noccaea_CQs.csv")
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

              
# %% Correlate CQs with n_pixel and mean plant CQ and relative area
df = pd.read_csv("data/Noccaea_CQs.csv")
df = df.loc[df.batch.notna(),:]

metals = ['Z', 'K', 'Ni', 'Ca']

dct = {"metal":[], "substructure":[], 
       "plant size correlation":[], 
       "mean metal concentration correlation":[],
       "substructure fractional area correlation":[]}


for i, metal in enumerate(metals):
    print(metal)
    subs_CQ = [ "metal_" + metal + "_" + name + "_CQ" for name in obj_class_lst[1:] ]
    # plt.figure(figsize=(10,10))
    print("Plant size correlation")
    for i,subs in enumerate(subs_CQ):
        r,p = pearsonr(df["metal_Z_plant_n_pix"], df[subs])
        print(subs, " r = ", r, " p = ", p)
        dct["metal"].append(metal)
        dct["substructure"].append(subs)
        dct["plant size correlation"].append(r)
        
        r,p = pearsonr(df["metal_" + metal + "_plant_meanC"], df[subs])
        dct["mean metal concentration correlation"].append(r)
        
        sbstrct_area = subs[:-2] + "n_pix"
        plant_area = "_".join(("metal", metal, "plant", "n_pix"))
        rel_strct_area = df[sbstrct_area] / df[plant_area] * 100        
        r,p = pearsonr(rel_strct_area, df[subs])
        dct["substructure fractional area correlation"].append(r)
 
df = pd.DataFrame(dct)
df.to_csv("data/output/article_images/plantsize_meanC_CQ_cors.csv")

# Scatter mean zinc concentration to plant size
# plt.scatter(df["metal_Z_plant_meanC"], df["metal_Z_plant_n_pix"])
# plt.title('scatter of mean zinc versus plant size')


# %% scatter substructure CQs against each other within metal
df = pd.read_csv("data/Noccaea_CQs.csv")
df["accession_str"] = df['Accession #'].astype(str)
df_noNAN = df.loc[df['batch'].notna(),:]
metals = ['Z', 'K', 'Ni', 'Ca']


def return_pval(x,y):
    return pearsonr(x, y)[1]

substructures = obj_class_lst[1:]
metal = "metal_Z"

fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()
for i, metal in enumerate(metals):
    pairplt_vars = [ "_".join(("metal", metal, name, "CQ")) for name in obj_class_lst[1:]]
    CQ_pair_corrs = df_noNAN[pairplt_vars].corr(method='pearson')
    CQ_pair_corrs.columns = substructures
    CQ_pair_corrs.index = substructures
    sns.heatmap(CQ_pair_corrs, annot=True, cbar=False, cmap="viridis", ax=axes[i])
    if metal == "Z":
        axes[i].set_title("Zn")
    else:
        axes[i].set_title(metal)
plt.savefig("data/output/article_images/CQ_paircorr.png", dpi=300)
plt.show()

# CQ_pair_cor_pval = df_noNAN[pairplt_vars].corr(method=return_pval)
# g = sns.pairplot(df_noNAN[pairplt_vars]) # , hue='accession_str', palette='bright'
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
assert os.path.exists("data/H2_CQ_table.csv", "H2_CQ_table.csv not found, did you run the main.R script?")

H2 = pd.read_csv("data/H2_CQ_table.csv", index_col=0)
df = pd.read_csv("data/Noccaea_CQs.csv")
df = df.loc[df["batch"].notna(),:]
random.seed(69) # Seed for random accessions
accessions = df["Accession #"].unique().tolist()
# Get every nth accession to end up with 25 "random" accessions
selected_accessions = accessions[0::len(accessions)//25] 

# Set up combined figure for raw data plant size, petiole CQ, rand CQ and H2-scores
fig, axes = plt.subplots(2,2, figsize=(20,10))
axes = axes.ravel()

# Show raw data supporting H2 for plant size, petiole CQ, rand CQ
toshow = ["metal_Z_plant_n_pix", "metal_Z_petiole_CQ", "metal_Z_rand_2_CQ"]
xlabs = ["plant size (pixels)", "petiole Zn CQ", "random substructure Zn CQ"]
acc_strs = [str(x) for x in selected_accessions]
for i, colname in enumerate(toshow):
    plt_df = df.loc[df["Accession #"].isin(selected_accessions),["Accession #", colname]]
    sns.stripplot(x="Accession #", y=colname, data=plt_df,
                jitter=False, color=sns.color_palette()[0],
                linewidth=1, ax=axes[i])# hue="substructure", palette=msk_hex_palette, legend=False)
    axes[i].set_ylabel(xlabs[i])

# Plot H2 for plant size, mean Zn, Zn sub CQs, random substructure CQ 
substructures = obj_class_lst[1:]
metals = ['Z', 'K', 'Ni', 'Ca']
metrics = ["plant_n_pix", "plant_meanC"] + substructures + ["rand_2"]
i_names = ["plant size", "mean [Zn]", "petiole Zn CQ", 
           "margin Zn CQ", "vein Zn CQ", "tissue Zn CQ", 
           "random \nsubstructure \nCQ"] 
colors = [np.array((105,105,105))/256, np.array((169,169,169))/256] + \
         [msk_hex_palette[s] for s in substructures] + \
         [np.array((119,136,153)) / 256] 


bars = ["_".join((m, "metal", metals[0])) for m in metrics]
data = H2.loc[bars, "H2_percent"]
data.index = i_names
sns.barplot(x=data.index, y=data.values, palette=colors, ax=axes[3])
plt.ylabel("H2 (%)")
plt.xticks(rotation=45)
plt.ylim((0,100))
plt.savefig("data/output/article_images/CQSizeMeanRand_H2.png", dpi=300, bbox_inches="tight")
plt.show()

xlabs = ["Zn", "K", "Ni", "Ca"]
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
# plt.savefig("data/output/article_images/CQ_H2_substrs.png", dpi=300, bbox_inches="tight")
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
# plt.savefig("data/output/article_images/CQ_H2_Noise_substrs.png", dpi=300, bbox_inches="tight")


# %% Get correlations between metals within substructures
df = pd.read_csv("data/Noccaea_CQs.csv")
df_noNAN = df.loc[df['batch'].notna(),:]

def return_pval(x,y):
    return pearsonr(x, y)[1]

substructures = obj_class_lst[1:]
metals = ["metal_Z", "metal_K", "metal_Ni", "metal_Ca"]
labs = ["Zn CQ", "K CQ", "Ni CQ", "Ca CQ"]
plt.figure(figsize=(10,10))
for i, substr in enumerate(substructures):
    plt.subplot(2,2,i+1)
    pairplt_vars = [ "_".join((metal, substr, "CQ")) for metal in metals]
    CQmetal_pair_corrs = df_noNAN[pairplt_vars].corr(method='pearson')
    CQmetal_pair_corrs.columns = metals
    CQmetal_pair_corrs.index = metals
    sns.heatmap(CQmetal_pair_corrs, annot=True, cbar=False, cmap="viridis",
                xticklabels=labs, yticklabels=labs)
    plt.title(substr)
plt.savefig("data/output/article_images/CQ_metalpaircorr.png", dpi=300)
plt.show()    


    # CQ_pair_cor_pval = df_noNAN[pairplt_vars].corr(method=return_pval)
    # g = sns.pairplot(df_noNAN[pairplt_vars]) # , 
    # plt.show()

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
precision = metrics.precision_score(Y_obs, Y_pred, labels=[1,2,3,4], average=None)
recall = metrics.recall_score(Y_obs, Y_pred, labels=[1,2,3,4], average=None)
print("f1")
print(F_scores)
print("precision")
print(precision)
print("recall")
print(recall)

# F_scores = []
# for i in range(len(para_df)):
#     Y_pred =  randpix_df["pred_class_" + str(i)]
#     conf_matrix = pd.DataFrame(metrics.confusion_matrix(Y_obs,Y_pred, [1,2,3,4]))
#     conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
#     sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True)
#     F_scores.append(metrics.f1_score(Y_obs,Y_pred, labels = [1,2,3,4], average='weighted'))
#     plt.show()


    
# %% Visualize sensitivity analysis
# TODO fix the manual blade size part in tandem with sensitivity analysis itself
from scripts.sensitivity_analysis import para_dct, para_map
from sklearn import metrics

sens_df = pd.read_csv("data/rand_pred_pixel_sens.csv", index_col=0, header=0)
para_df = pd.read_csv("data/sensitivity_paras.csv", index_col=0, header=0)


F1_dct = {"Parameter":[],"Parameter Value":[],"Substructure":[],
                              "recall":[], "precision":[], "F1":[]}

y_true = sens_df.obs_class
for k,v in para_map.items():
    colname = "".join(("pred_class_", str(k)))
    
    F1_scores = metrics.f1_score(y_true,sens_df[colname], labels = [1,2,3,4], average=None)
    precision = metrics.precision_score(y_true,sens_df[colname], labels=[1,2,3,4], average=None)
    recall = metrics.recall_score(y_true,sens_df[colname], labels=[1,2,3,4], average=None)
 
    F1_dct["Parameter"] += ["_".join((v.split("_")[0], v.split("_")[1]))] * len(F1_scores)
    F1_dct["Parameter Value"] += [int(v.split("_")[2])] * len(F1_scores)
    F1_dct["Substructure"] += obj_class_lst[1:]
    F1_dct["recall"] += recall.tolist()
    F1_dct["precision"] += precision.tolist()
    F1_dct["F1"] += F1_scores.tolist()

        
# Repeat for blade_ksize parameter value 15 (missing from para_map) 
colname = "pred_class"
parameter = "blade_ksize"
para_value = 15

F1_scores = metrics.f1_score(y_true,sens_df[colname], labels = [1,2,3,4], average=None)
precision = metrics.precision_score(y_true,sens_df[colname], labels=[1,2,3,4], average=None)
recall = metrics.recall_score(y_true,sens_df[colname], labels=[1,2,3,4], average=None)

F1_dct["Substructure"] += obj_class_lst[1:]
F1_dct["recall"] += recall.tolist()
F1_dct["precision"] += precision.tolist()
F1_dct["F1"] += F1_scores.tolist()
F1_dct["Parameter"] += [parameter] * 4
F1_dct["Parameter Value"] += [para_value] * 4

F1_df = pd.DataFrame(F1_dct)

# Plot F1 scores, accuracy and precision
def plot_scores(F1_df, score_name, xlabs):
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs = axs.ravel()
    for i, para in enumerate(F1_df.Parameter.unique().tolist()):
        sns.barplot(data=F1_df.loc[F1_df.Parameter == para,:],
            x="Parameter Value", y=score_name, hue="Substructure",
            palette=msk_hex_palette, ax=axs[i])
        xticks = F1_df.loc[F1_df.Parameter == para,"Parameter Value"].unique().tolist()
        xticks.sort()
        xticks = [str(tick) for tick in xticks]
        xticks[2] += "*"
        axs[i].set_xticks(range(len(xticks)))
        axs[i].set_xticklabels(xticks)
        axs[i].get_legend().remove()
        axs[i].set_xlabel(xlabs[para])
        axs[i].set_ylim(0.,1.0)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.8),loc = 'upper right')
    plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.25)

xlabs = {"blade_ksize":"blade opening kernel size [pixels]",
          "lap_ksize":"Laplacian kernel size [pixels]",
          "thin_th":"Threshold on Laplacian (thin) [-]",
          "fat_th": "Threshold on Laplacian (fat) [-]"}

plot_scores(F1_df, "F1", xlabs)
plt.savefig('data/output/article_images/F1_sensitivity.png', bbox_inches='tight')
plt.show()

plot_scores(F1_df, "precision", xlabs)
plt.savefig('data/output/article_images/precision_sensitivity.png', bbox_inches='tight')
plt.show()

plot_scores(F1_df, "recall", xlabs)
plt.savefig('data/output/article_images/recall_sensitivity.png', bbox_inches='tight')
plt.show()

# %% Inspect erronous pixels

ground_truth_path = "data/rand_pred_pixel.csv"
assert os.path.exists(ground_truth_path), "ground truth not found, make sure curated_pixels.py was run"

randpix_df = pd.read_csv(ground_truth_path, index_col=0, header=0)
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
        
    
# %% Check noise distribution
rev_msk_col_dct = {v:k for k,v in msk_col_dct.items()}
dct = {"plant_fn":[], "percentage":[], "noise_substruct":[]}
dct.update({sbstrct:[] for sbstrct in obj_class_lst[1:]})

for fn in plant_fns:
    for percentage in ['10', '20', '50', '75', '90','100']:
        
        multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn) # Load as RGB
        assert isinstance(multimsk, np.ndarray), "{} doesn't exsit".format(PLANT_MULTIMSK_PATH + fn)
        multimsk = multimsk.reshape(-1, multimsk.shape[2])
        
        noise_multimsk = cv2.imread("data/plant_noisemsk/" + percentage + "/" +  fn)
        assert isinstance(noise_multimsk, np.ndarray), "{} doesn't exsit".format("data/plant_noisemsk/" + percentage + "/" +  fn)
        noise_multimsk = noise_multimsk.reshape(-1, noise_multimsk.shape[2])
        
        for sbstrct in obj_class_lst[1:]:
            dct["plant_fn"].append(fn)
            dct["percentage"].append(percentage)
            dct["noise_substruct"].append(sbstrct)
            
            # Return indices where noised image equals certain substruct
            i_sub, _ = np.where(noise_multimsk == msk_col_dct[sbstrct])
                        
            # Count occurence of actual substruct under noised substruct pixels
            actual_under_noise = multimsk[i_sub, :]
            unique, counts = np.unique(actual_under_noise, return_counts=True, axis=0)
            unique = [tuple(arr) for arr in list(unique)]
            uc_dct = dict(zip(unique, counts))
            for actual_sbstrct in obj_class_lst[1:]:
                col = msk_col_dct[actual_sbstrct]
                try: # Find counts of actual substruct under noised mask
                    cnt = uc_dct[col]
                    dct[actual_sbstrct].append((cnt/counts.sum()) * 100)
                except KeyError: # Add 0 if no pixel counts for actual substruct
                    dct[actual_sbstrct].append(0)
                
# %% plot noise distributions
noise_dist_df = pd.DataFrame(dct)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,10))
axes = axes.ravel()
for i, perc in enumerate(['10', '20', '50', '75','90','100']):
    df = noise_dist_df.loc[noise_dist_df["percentage"] == perc,:]
    df_summed = df.groupby("noise_substruct").sum()
    df_perc = df_summed.apply(lambda x: x/x.sum() * 100, axis=1)
    df_perc.plot.bar(stacked=True, color=msk_hex_palette, ax=axes[i])
    # plt.xticks(range(4), obj_class_lst[1:])
    if i%3 ==0:
        axes[i].set_ylabel("actual pixels under noise (%)")
    axes[i].set_xlabel("Noised classes ({}% noised)".format(perc))
    
    axes[i].get_legend().remove()
plt.tight_layout()
plt.legend(title="actual pix under nosie", bbox_to_anchor=(2.1, 2.2),loc = 'upper right')
plt.savefig("data/output/article_images/noised_actual_dist.png", dpi=300, bbox_inches="tight")
noise_dist_df.to_csv("data/noise_actual_dist_df.csv")

