# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

class name():

    
    def __init__(self):
        
        self.value = 0

    def module1(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
        alpha = 0.3
        # img[img ==  0] = 1

        il_img = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        il_img = il_img.astype(np.float32)
        
        img = img.astype(np.float32)
        b,g,r = cv2.split(img)

        b = cv2.normalize(b, alpha=0.000000001,beta=1,norm_type =cv2.NORM_MINMAX)
        g = cv2.normalize(g, alpha=0.000000001,beta=1,norm_type =cv2.NORM_MINMAX)
        r = cv2.normalize(r, alpha=0.000000001,beta=1,norm_type =cv2.NORM_MINMAX)

        log_b = np.log(b)
        log_g = np.log(g)
        log_r = np.log(r)

        il_img = 0.5 + log_g + alpha * log_r - (1-alpha) * log_b 
        il_img = cv2.normalize(img,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

        cv2.imshow('output',il_img/np.amax(il_img))





        out = img
        return 0

    def module2(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        cv2.SuperpixelSEEDS.getNumberOfSuperpixels(img, num_iterations=4)
        out = img
        return out


################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    cls = name()

    img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')


    out = cls.module2(img)

    

    # cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    