# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
テンプレートマッチングとShape from Shadingを用いた岩検知
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class RockDetect():

    
    def __init__(self):
        
        self.value = 0

    def shadow_detect(self,img):
        '''
            処理の概要
            args : img     -> image,3ch
            dst  : shadow  -> 影領域が0の画像
            param:      -> 
        '''
        # 準備
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('img',img)   

        # ヒストグラムの描画        
        hist=img.ravel()
        plt.hist(hist,256,[0,256])
        plt.xlim([0,256])
        plt.pause(0.01)


        # 閾値処理
        img[img>35] = 255
        shadow = img

        # hist2=shadow.ravel()
        # plt.hist(hist2,256,[0,256])
        # plt.xlim([0,256])
        # plt.pause(0.01)

        # 画像表示
        cv2.imshow('img2',shadow)


        out = shadow
        return out

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

    RD = RockDetect()

    img = cv2.imread('../../image/rock/sol729.jpg')
    # img = cv2.imread('../../image/rock/'+sys.argv[1])
    if img is None:
        print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        sys.exit()
    img = cv2.resize(img,(512,512))


    out = RD.shadow_detect(img)

    

    # cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    