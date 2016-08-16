# -*- coding: utf-8 -*-
'''
特徴とTMの出力を受け取り、その領域が岩かどうか判断する。

    src: features   ->
         match_img  ->
    dst: new_features -> 岩じゃないものは削除した特徴の配列

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
        
        out = img
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


if __name__ == '__main__':

    cls = name()

    img = cv2.imread(sys.argv[1])
    # img = cv2.imread('../../image/rock/sol729.jpg')
    if img is None:
        print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        sys.exit()
    out = cls.module1(img)

    

    cv2.imshow('Input',  img)
    cv2.imshow('Output', out)
    cv2.waitKey(-1)
    