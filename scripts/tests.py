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


    a = [1,2,3,4]
    b = [5,6]
    c = [7,8]

    x, y = np.meshgrid(a,b)
    x = x.flatten()
    y = y.flatten()

    # print x
    # print y

    for i in range(8):
        print np.vstack((x[i],y[i]))


    # for p in ps:
        # print 'ok'
        # print p