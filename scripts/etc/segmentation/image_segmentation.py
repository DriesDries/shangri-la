# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from cv2 import cv

class ImageSegmentation():
  
    def __init__(self):
        
        self.value = 0

    def watershed(self,img):
        '''
            watershedで領域分割を行う
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gimg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv.CV_DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg) 

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

    def KMeanClustering(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = int(sys.argv[2])
        ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        cv2.imshow('res2',res2)

    def EdgeDetection(self,img):
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # gimg = cv2.GaussianBlur(gimg,(7,7),1)
        gimg = cv2.medianBlur(gimg,5)

        edges = cv2.Canny(gimg,200,80)
        cv2.imshow('edge',edges)

    def HarrisCorner(self,img):
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gimg, blockSize = 2, ksize = 3, k = 0.04)
        dst = cv2.dilate(dst,None)
        cv2.imshow('harris',dst/np.amax(dst))


################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    IS = ImageSegmentation()

    img = cv2.imread(sys.argv[1])
    # img = cv2.imread('../../planet_image/itokawa/itokawa2.png')
    # if img is None:
    #     print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
    #     sys.exit()


    # IS.watershed(img)
    IS.KMeanClustering(img)
    IS.EdgeDetection(img)
    IS.HarrisCorner(img)
    

    cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    