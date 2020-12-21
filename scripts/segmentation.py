# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:43:01 2020

@author: lucas
"""

# TODO rewrite code on generating random substructues and other random stuff (moved it from main to here)


import numpy as np
import itertools
import cv2
import pickle
import os
import random

from scripts import viz, draw

def divide_plants(RAW_TIFF_PATH, batch, polygon_dct, POLY_DCT_PATH, df, DF_SAVE_PATH):
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
        viz.plot_big(img_poly)
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
        return df
    
def create_foreground_masks(RAW_TIFF_PATH, batch, layers, BATCH_MSK_PATH):
    binary_mask_lst = []
    for layer in layers:
        img = cv2.imread(RAW_TIFF_PATH + batch + "- " + layer, 0)
        ret,th_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

def create_multimsks(batch, RAW_TIFF_PATH, BATCH_MSK_PATH, 
                     blade_ksize, lap_ksize, thin_th, fat_th,
                     msk_col_dct, path=False):
    
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
    large_contours = contouring(petiole, area_th = 0.00001) # Removes small misclassified petiole areas at blade edge
    petiole = contour2mask(petiole, large_contours)
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

    ## Create multi-color mask image
    multi_msk = np.zeros((msk.shape[0], msk.shape[1], 3))
    for name, partial_msk in msk_dct.items():
        col_BGR = msk_col_dct[name]
        partial_msk = partial_msk[:,:,None] # Add dimension for color
        multi_msk = np.where(partial_msk == 255, col_BGR, multi_msk)
        
    ## Save multimsk
    if path:
        cv2.imwrite(path + batch + "multimsk.tif", multi_msk)
    
    return multi_msk


# TODO this should be two or three separate functions
# TODO rewrite comments

def crop_plant(polygon, batch_img, fg_msk, bg_color = None):
    """Uses manually drawn polygons to divide batch images/masks into individual plant images/masks
    Input:
        polygon: manual polygon for 1 plant
        batch_img: image/mask of batches
        fg_msk: batch foreground mask
    Output:
        cropped image
    """
    # Black out all but plant in polygon
    if batch_img.ndim == 2:
        blacked_img = poly_crop(batch_img, polygon)
    elif batch_img.ndim == 3:
        blacked_img = poly_crop(batch_img, polygon, 
                                  col = (255,255,255), bg = bg_color) # msk_col_dct['background']
    
    # Crop image
    x,y,w,h = cv2.boundingRect(blacked_img)
    cropped_img = blacked_img[y:y+h,x:x+w]
    cropped_fg_msk = fg_msk[y:y+h,x:x+w]
    
    # Set background outside of plant foreground mask
    if cropped_img.ndim == 2:
        return np.where(cropped_fg_msk != 0, cropped_img, 0)
    elif cropped_img.ndim == 3:
        fg_msk_rgb = cv2.cvtColor(cropped_fg_msk * 255, cv2.COLOR_GRAY2RGB)
        return np.where(fg_msk_rgb == (255,255,255), cropped_img, bg_color)   
    

def make_individual_plant_images(POLYGON_DCT_PATH, batch, RAW_TIFF_PATH, 
                                 BATCH_MSK_PATH, multimsk, RAW_TXT_PATH, 
                                 PLANT_IMG_PATH, PLANT_MSK_PATH, PLANT_MULTIMSK_PATH,
                                 metal, msk_col_dct, create_masks = False):
    # Load the polygon coordinates
    with open(POLYGON_DCT_PATH, "rb") as f:
        polygon_dct = pickle.load(f)
        
    img_path = RAW_TIFF_PATH + batch + "- Image.tif"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), "{} doesn't exsit".format(img_path)
    
    msk_path = BATCH_MSK_PATH + batch + "batchmsk.tif"
    msk = cv2.imread(msk_path,  cv2.IMREAD_GRAYSCALE) // 255 # load image as binary
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(msk_path)
    
    if type(multimsk) == str:
        multimsk_path = multimsk + batch + "multimsk.tif"
        multimsk = cv2.imread(multimsk_path) # Load as RGB
    assert isinstance(msk, np.ndarray), "{} doesn't exsit".format(multimsk_path)
        
    # Zimg_path = RAW_TXT_PATH + batch + "- Zn.txt"
    # Zimg = np.loadtxt(Zimg_path, delimiter=",", skiprows=1)
    #TODO harmonize folder names, change Zimg to Znimg
    if metal == "Z":
        metal_raw_path = "Zn"
    else:
        metal_raw_path = metal
        
    metalimg_path = RAW_TXT_PATH + batch + "- " + metal_raw_path + ".txt"    
    metalimg = np.loadtxt(metalimg_path, delimiter=",", skiprows=1)
    
    # Dilate mask to include a strip of background around the plant Zimage
    kernel = np.ones((5,5),np.uint8)
    dil_msk = cv2.dilate(msk, kernel, iterations = 1)
    
    # Loop over polygon dictionaries
    for acc_rep, polygon in polygon_dct[batch].items():
        # Create mask/image name
        accession, replicate = acc_rep.split("_")
        fn = "_".join([batch, accession, replicate])
        
        # Crop img, Zimg, msk and multimsk using polygon
        blacked_img = poly_crop(img, polygon)
        blacked_metalimg = poly_crop(metalimg, polygon)
        blacked_msk = poly_crop(msk, polygon)
        bged_multimsk = poly_crop(multimsk, polygon, 
                                             col = (255,255,255), bg = msk_col_dct['background'])
            
        # Crop image to bounding box around polygon
        x,y,w,h = cv2.boundingRect(blacked_img)
        plant_dil_msk = dil_msk[y:y+h,x:x+w]
        
        if create_masks:
            plant_msk = blacked_msk[y:y+h,x:x+w] * 255
            
            dirty_plant_img = blacked_img[y:y+h,x:x+w]
            dirty_plant_multimsk = bged_multimsk[y:y+h,x:x+w]
            
            plant_img = np.where(plant_dil_msk == 1, dirty_plant_img, 0)
            plant_dil_mskRGB = cv2.cvtColor(plant_dil_msk * 255, cv2.COLOR_GRAY2RGB)
            plant_multimsk = np.where(plant_dil_mskRGB == (255,255,255), dirty_plant_multimsk, msk_col_dct['background'])
        if create_masks == "multi":
            plant_dil_mskRGB = cv2.cvtColor(plant_dil_msk * 255, cv2.COLOR_GRAY2RGB)
            plant_multimsk = np.where(plant_dil_mskRGB == (255,255,255), dirty_plant_multimsk, msk_col_dct['background'])
            # cv2.imwrite(PLANT_IMG_PATH + fn + ".tif", plant_img)
            # cv2.imwrite(PLANT_MSK_PATH + fn + ".tif", plant_msk)
            # cv2.imwrite(PLANT_MULTIMSK_PATH + fn + ".tif", plant_multimsk)
        
        dirty_plant_metalimg = blacked_metalimg[y:y+h,x:x+w]
        plant_metalimg = np.where(plant_dil_msk == 1, dirty_plant_metalimg, 0)
        # np.savetxt("data/plant_" + metal + "img/" + fn + ".txt", plant_metalimg, fmt='%f', delimiter=",")



def create_noised_msks(PLANT_MULTIMSK_PATH, PLANT_MSK_PATH, plant_fns, msk_col_dct, percentage):
    # Check for folder existence and create
    fraction = percentage / 100
    if not os.path.exists("data/plant_noisemsk/" + str(percentage)):
        os.makedirs("data/plant_noisemsk/" + str(percentage))

    # Load masks, inject noise and write to disk
    for fn in plant_fns:
        multimsk = cv2.imread(PLANT_MULTIMSK_PATH + fn)
        msk = cv2.imread(PLANT_MSK_PATH + fn, cv2.IMREAD_GRAYSCALE)
        noised_msk = multimsk.copy()
            
        x,y = np.where(msk == 255)
        n_pixels = int(len(x) * fraction)
        
        i_lst = random.sample(range(len(x)), n_pixels) 
        cols = np.array([col for k, col in msk_col_dct.items() if k != "background"])
        col_lst = cols[np.random.choice(len(cols), size = n_pixels)]
        noised_msk[x[i_lst], y[i_lst]] = col_lst
        
        cv2.imwrite("data/plant_noisemsk/" + str(percentage) + "/" + fn, noised_msk)
        
def create_rand_substructure(PLANT_MSK_PATH, PLANT_RANDMSK_PATH, N_pixels):
    if not os.path.exists(PLANT_RANDMSK_PATH + str(N_pixels)):
        os.makedirs(PLANT_RANDMSK_PATH + str(N_pixels))
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
        
        # Dilate pixels
        kernel = np.ones((30,30),np.uint8)
        dil_msk = cv2.dilate(rand_msk, kernel, iterations = 1)
        
        # Crop to plant mask
        dil_msk = dil_msk & msk
        dil_msk = dil_msk * 255
        
        cv2.imwrite(PLANT_RANDMSK_PATH + str(N_pixels) + "/" + fn,  dil_msk)
        
# Helper functions
def contouring(img, area_th = 0.0001):
    """Use open-cv contouring to get contours of a binary image.
    
    This function calculates all contours but will return only the contours
    larger than area threshold fraction of the image to avoid noise.
    Input: img, a numpy array containing a binary image.
    Returns: large_contours, open-cv contours larger than 1 percent of the image.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    small_contours = []
    x , y = np.shape(img)
    total_area = x * y
    for c in contours:
        if cv2.contourArea(c) > int(total_area * area_th):
            large_contours.append(c)
        else:
            small_contours.append(c)
    large_contours = np.array(large_contours)
    return large_contours

def contour2mask(img, contours):
    """This function will create a mask based on the contours and the original image.
    
    Input: img, a numpy array with an image.
    Input: contours, open-cv contours.
    Returns: binary_mask, the contours drawn on a black image. Not actually binary since black is 0 and white is 255.
    """
    binary_mask = np.zeros(np.shape(img), dtype=np.uint8)
    cv2.drawContours(binary_mask, contours, -1, (255,255,255), -1)
    return binary_mask
	
def poly_crop(img, polygon, col = 255, bg = 0):
    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, polygon, col)
    res = np.where(stencil == 255, img, bg)
    return res
