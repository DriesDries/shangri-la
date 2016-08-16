 # -*- coding: utf-8 -*-
'''
    Test function
    
    Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def chi_square_distance():
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    from scipy import stats
    
    a = [72, 23, 16, 49]
    b = [40, 40, 40, 40]

    dis = stats.chisquare(a, b)
    print dis[0] # カイ二乗距離
    print dis[1]



if __name__ == '__main__':

    # chi_square_distance()

    img = cv2.imread('../../data/g-t_data/original/spirit050-2.jpg')
    img = img[400:800,400:800]
    # cv2.imshow('img',img)
    # cv2.waitKey(-1)
    cv2.imwrite('../../data/g-t_data/resized/spirit050-2.png',img,[int(cv2.IMWRITE_JPEG_QUALITY), 0])