# -*- coding: utf-8 -*-
'''
小さい岩領域と大きい岩領域の結合
Usage: $ python merge.py <argv>
''' 

import cv2
import numpy as np

def main(img1,img2):
    ror = merge(img1,img2)
    return ror

def merge(img1,img2):
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    # もしcolorだったらgrayに
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    ror = np.zeros_like(img1)
    
    ror[img1!=0] = 255
    ror[img2!=0] = 255

    cv2.imshow('img',ror)

    return ror



if __name__ == '__main__':

    img1 = cv2.imread('../../../image/sample/sample1.png')
    img2 = cv2.imread('../../../image/sample/sample2.png')
    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')

    main(img1,img2)

    cv2.waitKey(-1)
    