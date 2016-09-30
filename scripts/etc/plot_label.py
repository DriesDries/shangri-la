# -*- coding: utf-8 -*-
"""
===================
Label image regions
===================

This example shows how to segment an image with image labelling. The following
steps are applied:

1. Thresholding with automatic Otsu method
2. Close small holes with binary closing
3. Remove artifacts touching image border
4. Measure image regions to filter small objects

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb


image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image) # 閾値が出る
bw = closing(image > thresh, square(3)) # True or False


# remove artifacts connected to image border 
cleared = bw.copy()
clear_border(cleared) #画像の端に繋がってるのは削除している

# label image region
label_image = label(cleared) # 領域の二値化画像みたいなのを生成、領域数もカウントしてる
image_label_overlay = label2rgb(label_image, image=image) #いらないとこをマスクで消したカラー画像


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image_label_overlay)

print label_image


for region in regionprops(label_image):

    print region.area

    # skip small images
    if region.area < 100:
        continue

    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

plt.show()
cv2.waitKey(-1)