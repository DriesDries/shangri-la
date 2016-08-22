# -*- coding: utf-8 -*-
'''

    Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def module1(img):
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    
    out = img
    return out

def module2(img):
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''

    out = img
    return out


if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])
    # img = cv2.imread('../../image/rock/sol729.jpg')
    if img is None:
        print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        sys.exit()
    out = module1(img)

    cv2.imshow('Input',  img)
    cv2.imshow('Output', out)
    cv2.waitKey(-1)
    