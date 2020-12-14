# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:11:18 2020

@author: lucas
"""
# %% create mask r-cnn dictionary and images
# TODO first got id "0" this should be reserved for bg right?
# TODO; this was originally in main, needs main things

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
    
    
    
    