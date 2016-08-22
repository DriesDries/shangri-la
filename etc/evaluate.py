# -*- coding: utf-8 -*-
'''
    アルゴリズムに対して評価を行うスクリプト

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

def display_result():
        ## Display result
    e = 0.2         # test
    plt.bar(0, precision, align = "center", yerr = e, ecolor = "black")
    plt.bar(2, recall,  align = "center", yerr = e, ecolor = "black")
    plt.xticks([0,2], ['Precision','Recall'])
    plt.xlim(-2,4)
    plt.ylim(0,1)
    plt.ylabel('precision or recall')
    plt.title('Target selection performance')

def img2list(img):
    ''' 
    画像で非0の座標をlistに

    '''
    seed_list = []
    value = []

    for i in range(len(img)):
        for j in range(len(img)):
            if img[j,i] != 0:
                seed_list.append([j,i])
                value.append(img[j,i])
    
    return seed_list

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


    return true_ror.astype(np.uint8), labels, num, rock, soil

def get_recall(ror_true, seed_list, num, labels, seed_img):
    '''
        recall = (検出できた岩の数) / (全ての岩の数)

        それぞれの岩領域の中にひとつでも種があれば、検出できたこととする。

        detect_img -> 検出できた岩は255、できなかった岩は150とした画像


    '''


    non = np.zeros_like(ror_true)
    reg = non.copy()
    seed = seed_img.copy()

    detect = 0
    nondetect = 0
    detect_img = non.copy()

    for i in range(1,num):
        
        ## 領域のみに処理する
        reg[labels==i] = 255 # 岩が255
        seed[labels!=i] = 0  # 岩以外の種が0

        ## 検出できてるとき
        if np.array(np.nonzero(seed)).shape[1] != 0:
            detect += 1
            detect_img[labels==i] = 255

        ## 検出できてないとき
        elif np.array(np.nonzero(seed)).shape[1] == 0:
            nondetect += 1
            detect_img[labels==i] = 150

        ## 初期化と描画
        reg[reg!=0] = 0 
        seed = seed_img.copy()

    recall = 1. * detect/num

    # cv2.imshow('detect',detect_img)
    # print 'total = {}, detected = {}, NOTdetected = {}'.format(num-1,detect,nondetect)
    # print 'recall = {}'.format(recall)

    return recall, detect

def get_precision(ror_true, seed_list, num, labels, seed_img):
    '''
        seedのprecisionとrecallを求める
        
        Precision = (岩だった数)/(検出した岩の数)
        
        ror_true -> 岩領域の二値画像
        num      -> 全部の岩の数

        ちょっと定義違うかも
    '''
    detect = 0

    ## それぞれの座標が入ってるかどうか
    for seed in seed_list:
        if ror_true[tuple(seed)] != 0:
            detect += 1

    precision = 1. * detect / len(seed_list)

    return precision

def eval_temp(img):
    '''
        forで回してscoreをとってくる
        meshgrid
    '''
    parameters = []

    for param in parameters:


if __name__ == '__main__':

    ## Ground-Truth data acquisition
    img       = cv2.imread('../../data/g-t_data/resized/spirit118-1.png')
    label_img = cv2.imread('../../data/g-t_data/label/spirit118-1.png')
    # cv2.imwrite('../../data/g-t_data/rock_region/spirit118-1.png',rock_region,[int(cv2.IMWRITE_JPEG_QUALITY), 0])
    # cv2.imwrite('../../data/g-t_data/soil_region/spirit118-1.png',soil_region,[int(cv2.IMWRITE_JPEG_QUALITY), 0])
    
    ## Extract features of GT-data
    true_ror, labels, true_num, rock_region, soil_region = label(img,label_img)
    
    ## Predicted data acquisition
    seed_img  = cv2.imread('../../data/seed_img.png',0)
    seed_list = img2list(seed_img)

    ## Template Matching analysis
    eval_temp(img)
    recall, pred_num = get_recall(true_ror, seed_list, true_num, labels, seed_img)
    precision =     get_precision(true_ror, seed_list, true_num, labels, seed_img)

    ## Converage analysis
    # get_convarage()

    ## Range analysis
    # get_range_performance()

    ## Size analysis
    # get_size_perfomance()


    # cv2.imshow('img',img)

    ## Display results
    print '======================='
    print 'TRUE Rock     = {}'.format(true_num)
    print 'DETECTED Rock = {}'.format(pred_num)
    print 'Precision     = {}'.format(round(precision, 3))
    print 'Recall        = {}'.format(round(recall   , 3))
    print '======================='

    # plt.show()
    cv2.waitKey(-1)