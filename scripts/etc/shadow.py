# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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


        # adap_img = cv2.adaptiveThreshold(img,100,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,(9,9))

        g_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('g_img',g_img)
        adap_img = cv2.adaptiveThreshold(g_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=99, C=30)
        adap_img = 255 - adap_img
        cv2.imshow('adap',adap_img)

        print adap_img.shape

        self.display_histgram(img)
        cv2.rectangle(img,(0,0),(511,100),0,-1)



        _,thresh_img = cv2.threshold(img,30,255,1)
        _,thresh_img2 = cv2.threshold(img,150,255,0)

        shadow = cv2.cvtColor(thresh_img,cv2.COLOR_BGR2GRAY) #非0がそれぞれの領域
        rock = cv2.cvtColor(thresh_img2,cv2.COLOR_BGR2GRAY)



        # result = np.zeros_like(img)
        b,g,r = cv2.split(img)
        # r[rock != 0] = 255
        r[adap_img != 0] = 150
        result = cv2.merge((b,g,r))        
        cv2.imshow('result',result)



        # cv2.imshow('img2',thresh_img2)
        self.display_histgram(img)
        out = adap_img
        return out

    def display_histgram(self,img):
        '''
            処理の概要
            args :      -> 1ch or 3ch
            dst  :      -> 
            param:      -> 
        '''
        hist=img.ravel()
        plt.hist(hist,256,[0,256])
        plt.xlim([0,256])
        plt.ylim([0,40000])
        plt.pause(0.01)



            

    def module2(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        out = img
        return out


################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    cls = name()

    img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')
    img2 = cv2.imread('../../image/rock/dog0_screenshot_07.06.2016.png')

    # img = cv2.resize(cv2.imread('../../image/rock/spiritsol118navcam.jpg'),(512,512))

    out = cls.module1(img)

    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    _,img2 = cv2.threshold(img2,20,255,0)

    b,g,r = cv2.split(img)
    r[out != 0] = 200
    g[img2 != 0] = 200
    b[out != 0] = 0

    img2 = cv2.merge((b,g,r))

    cv2.imshow('Input',  img)
    cv2.imshow('Output', img2)
    cv2.waitKey(-1)
    