# -*- coding: utf-8 -*-
'''
    入力画像をSuper Pixelに分割する
    Usage : $ python superpixel.py
'''
import numpy as np
import cv2

from skimage import segmentation as seg
from skimage import util
from skimage import io
import matplotlib.pyplot as plt
 
def super_pixel(img):
    

    fig = plt.figure("Superpixels -- segments",figsize=(16,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # loop over the number of segments
    for numSegments in [500,1000]:
        # apply SLIC and extract (approximately) the supplied number of segments
        segments = seg.slic(img, n_segments = numSegments, sigma = 2)
        
        sp = seg.mark_boundaries(img, segments, color=(1,1,0), outline_color=None, mode='outer')
        
        if numSegments == 500:
            ax1.imshow(sp)
        if numSegments == 1000:
            ax2.imshow(sp)        

    plt.pause(-1)    


if __name__ == '__main__':

    img = util.img_as_float(io.imread('../../../data/g-t_data/resized/spirit118-1.png'))
    

    super_pixel(img)
    
