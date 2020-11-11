# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:07:35 2020

@author: lucas
"""
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

# %%

# %% Get vein mask
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