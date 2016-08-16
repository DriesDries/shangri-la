# -*- coding: utf-8 -*-
'''

    Usage: $ python template.py <argv>

    ToDo:
        ・繋がってるとこは緑で境界を書く
        ・precisionの表示方法変える
        ・TMの値変えたときのpresicionとrecall
            - textureで種消したらその変化もプロットする
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import main
import skimage.measure as sk

def label(img, label_img):
    '''
        label画像から岩領域を抽出する
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''

    ## Binary image of Region of Rocks
    size = label_img.shape[0]
    true_ror = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # Soil
            if label_img[i,j,0] == label_img[i,j,1] == label_img[i,j,2]:
                pass
            # Rock
            else:
                true_ror[i,j] = 255

    ## Get label
    labels, num = sk.label(true_ror, return_num = True)

    ## Get regions of rock and soil
    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rock = gimg.copy()
    soil = gimg.copy()
    rock[true_ror==0] = 0 
    soil[true_ror!=0] = 0


    return true_ror, labels, num, rock, soil

def get_seed_score(ror_true, seed_list, num):
    '''
        seedのprecisionとrecallを求める
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    correct = 0

    ## それぞれの座標が入ってるかどうか
    for seed in seed_list:
        if ror_true[tuple(seed)] != 0:
            correct += 1

    ## Calculate score # 定義が違う # パラメータ変更してったときの値を配列にいれる
    precision = correct / num
    # recall = 

    ## Display result
    precision = 0.7 # test
    recall = 0.4    # test
    e = 0.2         # test
    plt.bar(0, precision, align = "center", yerr = e, ecolor = "black")
    plt.xticks([0], ['Proposed'])
    plt.xlim(-3,3)
    plt.ylim(0,1)
    plt.ylabel('precision')
    plt.title('Target selection performance')

    return precision, recall, correct

def get_pc(ror_true):
    pass



if __name__ == '__main__':

    ## Original image acquisition
    img = cv2.imread('../../data/g-t_data/resized/spirit118-1.png')

    ## Ground-Truth data acquisition
    label_img = cv2.imread('../../data/g-t_data/label/spirit118-1.png')
    true_ror, labels, true_num, rock_region, soil_region = label(img,label_img)
    # feature_true = 

    ## Predicted data acquisition
    # pred_ror = cv2.imread('')
    # pred_seed = cv2.imread('')
    seed_list = [[1,2,],[3,4],[5,6]]

    ## Template Marching analysis
    precision, recall, pred_num = get_seed_score(true_ror, seed_list, true_num)
    # 値変えたときのpresicionとrecall

    ## Texture analysis

    ## Display results
    print '======================='
    print 'TRUE Rock Number = {}'.format(true_num)
    print 'PRED Rock Number = {}'.format(pred_num)
    print 'Precision        = {}'.format(precision)
    print 'Recall           = {}'.format(recall)
    print '======================='

    cv2.imshow('img',img)
    # plt.show()
    plt.pause(1)
    cv2.waitKey(-1)