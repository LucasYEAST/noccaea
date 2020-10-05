# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:40:58 2020

@author: lucas
"""
from scripts import utils, processing
from scripts.viz import *
import itertools

import cv2
import numpy as np
import pandas as pd

RAW_TIFF_PATH = "data/raw_data/tiff/"
BATCH_MSK_PATH = "data/batch_msk/"
BATCH_MULTIMSK_PATH = "data/batch_multimsk/"

PLANT_IMG_PATH = "data/plant_img/"
PLANT_MSK_PATH = "data/plant_msk/"
PLANT_MULTIMSK_PATH = "data/plant_multimsk/"

batchname_lst = utils.get_batch_names(RAW_TIFF_PATH)

obj_class_lst = ["background", "petiole", "margin", "vein", "tissue"]
msk_col_dct = get_colors(obj_class_lst)


# %% 1. Create batch foreground masks
layers = ["Ca.tif", "K.tif", "Ni.tif", "Image.tif"]
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
    mask = (binary_mask_lst[0] == 255) | (binary_mask_lst[1] == 255) | (binary_mask_lst[2] == 255) \
            | (binary_mask_lst[3] == 255)
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
#     bg_overlay = cv2.addWeighted(img, alpha, background.astype("uint8"), beta, 0.0)
#     plot_big(bg_overlay[:300,:300])
    
    # Get blade by opening on the whole plant mask with a large kernel to remove the petiole and artefacts
    blade_kernel = np.ones((15,15),np.uint8)
    blade = cv2.morphologyEx(msk, cv2.MORPH_OPEN, blade_kernel)
    blade = np.where((blade == 255) & (msk == 255), 255, 0) # Opening adds some pixels outside mask I beleive
    blade = blade.astype("uint8")
#     blade_overlay = cv2.addWeighted(img, alpha, blade, beta, 0.0)

    # Now get the petiole masks by subtracting the blade from the whole plant mask 
    # followed by another smaller kernel opening
    petiole = np.where(msk != blade, msk, 0)
    pet_kernel = np.ones((3,3), np.uint8)
    petiole = cv2.morphologyEx(petiole, cv2.MORPH_OPEN, pet_kernel)
    msk_dct["petiole"] = petiole
    # plot_big(petiole)


    ## Get leaf margin
    margin_kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(blade, cv2.MORPH_GRADIENT, margin_kernel)
    margin = np.where((blade == 255) & (gradient == 255), 255, 0).astype("uint8")
    msk_dct["margin"] = margin
#     margin_overlay = cv2.addWeighted(img, alpha, margin, beta, 0.0)
#     plot_big(margin_overlay[:300,:300])
    
    ## Get vein mask
    blade_img = np.where(blade, img, 0)
    neg_veins = cv2.Laplacian(blade_img,cv2.CV_64F, ksize=7)
    # neg_vein_mask = np.where(neg_veins > 0, 255, 0)
    # vein_mask = np.where((neg_vein_mask == 0) & (blade == 255), 255, 0)
    
    neg_vein_mask_th = np.where(neg_veins < -1000, 255, 0)
    vein_mask_th = np.where((neg_vein_mask_th == 255) & (blade == 255), 255, 0)
    
    marginless_veins = np.where((vein_mask_th == 255) & (margin == 0), 255, 0)
    msk_dct['vein'] = marginless_veins
    
    # blade_img = cv2.GaussianBlur(blade_img,(5,5), sigmaX=0)
    # th_veins = cv2.adaptiveThreshold(blade_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,7,0)
    # plt.figure(figsize=(25,25))
    # plt.subplot(2,2,1),plt.imshow(th_veins[:300,:300],'gray')
    # plt.subplot(2,2,2),plt.imshow(img[:300,:300], 'gray')
    # plt.subplot(2,2,3),plt.imshow(vein_mask[:300,:300], 'gray')
    # plt.subplot(2,2,4),plt.imshow(vein_mask_th[:300,:300], 'gray')
    # plt.show()
    
    ## get leaf tissue
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
    
    # msk_rgb = cv2.cvtColor(msk, cv2.COLOR_GRAY2RGB)
    # plot_big2(multi_msk, msk_rgb)

    # msk_rgb = np.where(msk_rgb == (255,255,255), (255,30,30), msk_rgb)
    # overlay(multi_msk, msk_rgb)
    # multi_msk = cv2.cvtColor(multi_msk.astype("uint8"), cv2.COLOR_RGB2BGR)
    cv2.imwrite(BATCH_MULTIMSK_PATH + batch + "multimsk.tif",  multi_msk.astype("uint8"))

# %% 3. Manually draw individual plant masks
# drawing = False	 # true if mouse is pressed 
# mode = True		 # if True, draw rectangle. 
# ix, iy = -1, -1

# # Load plant data

# # Create column for bbox coordinates

# # Iterate over batches
# for batch in batchname_lst:
#   # Open batch image
    
#   # Iterate over plants in batch
  
#   # Display accession and repetition; instruct to draw bounding box
  
#   # Store bounding box coords in dataframe
  
#   # save/overwrite dataframe on disk
    
# img = np.zeros((512, 512, 3), np.uint8) 
# cv2.namedWindow('image') 
# cv2.setMouseCallback('image', draw_circle) 

# while(1): 
# 	cv2.imshow('image', img) 
# 	k = cv2.waitKey(1) & 0xFF
# 	if k == ord('m'): 
# 		mode = not mode 
# 	elif k == 27: 
# 		break

# cv2.destroyAllWindows() 



# %% 3. Automatically create individual plant masks from batch masks

df = pd.read_excel("data/Noccaea_raw.xlsx", sheet_name = "ALL data")
DF_SAVE_PATH = "data/Noccaea_stats.csv"

kernel = np.ones((5,5),np.uint8)
for batch in batchname_lst:
    img_path = RAW_TIFF_PATH + batch + "- Image.tif"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(img_path)
    
    msk_path = BATCH_MSK_PATH + batch + "batchmsk.tif"
    msk = cv2.imread(msk_path,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(msk_path)
    
    multimsk_path = BATCH_MULTIMSK_PATH + batch + "multimsk.tif"
    multimsk = cv2.imread(multimsk_path) # Load as RGB
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(multimsk_path)
    
    # Create individual plant masks
    contours = processing.contouring(msk)
    filled_contours =  [processing.create_mask(msk,[x]) for x in contours]
    
    # Find and combine overlapping plant masks #TODO increase speed check part 2 with itertools thing
    new_sets = []
    overlap_lst = []
    # import time
    # a = time.time()
    delete_lst = []
    for i in range(len(filled_contours) - 1):
        # overlap = [i]
        for j in range(i + 1, len(filled_contours)):
            added_img = cv2.add(filled_contours[i], filled_contours[j], dtype=cv2.CV_8U) # Saturated add
            if added_img.sum() < (filled_contours[i].sum() + filled_contours[j].sum()): # filled contours overlap, throw out smaller one
                if filled_contours[i].sum() > filled_contours[j].sum():
                    delete_lst.append(j)
                else:
                    delete_lst.append(i)
    delete_lst = list(set(delete_lst))
    for index in sorted(delete_lst, reverse=True):
        del filled_contours[index]
    
    # overlap.append(j)
        # overlap_lst.append(overlap) # list of overlap lists
    # print(time.time() - a)
    # plant_parts_lst = utils.combine_plant_parts(overlap_lst)

    # Combine plantmasks and bboxes from parts
    # plant_lst = []
    # for plant in plant_parts_lst:
    #     connected_plant = np.zeros(msk.shape)
    #     for part_i in plant:
    #         connected_plant =  cv2.add(connected_plant, filled_contours[part_i], dtype=cv2.CV_8U) # "saturated add"; if > 255 -> 255
    #     plant_lst.append(connected_plant)
   
    for i, plant in enumerate(filled_contours):
        # Dilate plant masks a bit to make sure the whole plant is in the bbox
        # plant = cv2.dilate(plant, kernel, iterations = 1)
        x,y,w,h = cv2.boundingRect(plant)
        x,y = max(x - 10, 0), max(y - 10, 0) # Expand bbox by 10 unless bbox crosses image boundary
        w,h = min(w + 20, msk.shape[1] - x), min(h + 20, msk.shape[0] - y) #Expand bbox by 10 unless bbox crosses image boundary
                
        ## Use bounding box coords to create new array
        dirty_plant_img = img[y:y+h,x:x+w] # Crop image using bounding box (contains parts of other plants)
        dirty_plant_msk = msk[y:y+h,x:x+w] # Crop plant mask using bounding box (contaisn part of other plants)
        dirty_plant_multimsk = multimsk[y:y+h,x:x+w]
        clean_plant_msk = plant[y:y+h,x:x+w]
        
        ## Document accession and replicate
        plot_big2(plant, img)
        confirm = False
        while confirm == False:
            accesion = input("tell me the accession")
            replicate = input("tell me the replicate")
            confirmation = input("accesion: " + accesion + " replicate: " + replicate + " confirm: y/n")
            if confirmation == "y":
                confirm = True
        
        if accesion == False:
            

        fn = "_".join([batch, accession, replicate])
        # Save batch, bbox, fn
            
        df.loc[(df["Accession #"] == accesion) & (df["Biological replicate"] == replicate) & 
               (df["Plant part"] == "shoot"), ["batch", "bbox", "fn"]] = [batch, (x,y,w,h), fn]
                
        
        
        ## Create tight fitting mask
        # blur = cv2.GaussianBlur(dirty_plant_img,(3,3),0)
        # th_img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,301,0)
        # ret,th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # 
        # th_img = cv2.Laplacian(dirty_plant_img,cv2.CV_64F, ksize=15)
        # th_img = cv2.Canny(dirty_plant_img,150,200)
        # plot_big(th_img)
        
        # con = processing.contouring(th_img.astype("uint8"), area_th = 0.01)
        # tight_plant_msk = processing.create_mask(dirty_plant_img, con)
        
        # ## Remove plant parts at image borders
        plant_msk = np.where(clean_plant_msk == 255, dirty_plant_msk, 0)
        plant_img = np.where(clean_plant_msk == 255, dirty_plant_img, 0)
        clean_plant_msk_gray = cv2.cvtColor(clean_plant_msk, cv2.COLOR_GRAY2RGB)
        plant_multimsk = np.where(clean_plant_msk_gray == 255, dirty_plant_multimsk, msk_col_dct['background'])
        
        cv2.imwrite(PLANT_IMG_PATH + batch + " " + str(i) + ".tif", plant_img)
        cv2.imwrite(PLANT_MSK_PATH + batch + " " + str(i) + ".tif", plant_msk * 255)
        cv2.imwrite(PLANT_MULTIMSK_PATH + batch + " " + str(i) + ".tif", plant_multimsk)

    
# %% 4. Calculate concentration statistics
for fn in os.listdir(PLANT_IMG_PATH):
    img = 


