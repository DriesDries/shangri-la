# -*- coding: utf-8 -*-
'''
    template matchingのseedに関して評価を行う

    Usage: $ python tm_eval.py <argv>

    ToDo:

        ・ meshgridでtuningできるように
        ・ 距離、大きさに分けて表示
        
        ・ textureで種消したらその変化もプロットする
        ・ textureで種消したらその変化もプロットする
        
''' 
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import skimage.measure as sk

def main(seed_img, ror_true, label_img, true_features):

    recall, pred_num, detect_img, true_features = get_recall(seed_img, label_img, true_features)
    precision = get_precision(ror_true, label_img, seed_img)
    # detect_rock = get_detect_img(img, detect_img)
    
    return precision, recall, np.array(true_features), detect_img.astype(np.uint8)


def get_recall(seed_img, labels, true_features):
    '''
        recall = (検出できた岩の数) / (全ての岩の数)

        それぞれの岩領域の中にひとつでも種があれば、検出できたこととする。

        detect_img -> 検出できた岩は255、できなかった岩は150とした画像

    '''
    num = np.amax(labels)
    non = np.zeros_like(labels)
    reg = non.copy()
    seed = seed_img.copy()

    detect = 0
    nondetect = 0

    detect_img = non.copy()

    for i in range(1, num):
        
        ## 領域のみに処理する
        reg[labels==i] = 255 # 岩が255
        seed[labels!=i] = 0  # 岩以外の種が0
        
        ## 検出できてるとき
        if np.array(np.nonzero(seed)).shape[1] != 0:
            detect += 1
            detect_img[labels==i] = 255
            true_features[i][0]= np.array(np.nonzero(seed)).shape[1]

            ## ここで領域の大きさに応じて追加する？
            ##  iとpsizeを追加する

        ## 検出できてないとき
        elif np.array(np.nonzero(seed)).shape[1] == 0:
            nondetect += 1
            detect_img[labels==i] = 150

        ## 初期化と描画
        reg[reg!=0] = 0 

        seed = seed_img.copy()

    recall = 1. * detect/num

    return recall, detect, detect_img, true_features


def get_precision(ror_true, labels, seed_img):
    '''
        precisionを求める
        それぞれの種が岩領域の中に入っているかどうか
        
        Precision = (岩だった数)/(検出した岩の数)
        
        ror_true -> 岩領域の二値画像
        num      -> 全部の岩の数
    '''
    detect = 0

    seed_list = np.nonzero(seed_img)

    for y,x in zip(seed_list[0], seed_list[1]):

        if ror_true[y,x] != 0:
            detect += 1

    precision = 1. * detect / len(seed_list[0])

    return precision

def main2(ror_true, seed_list, true_num, labels, seed_img):
    recall, pred_num, detect_img = get_recall(ror_true, seed_list, true_num, labels, seed_img)
    # precision =     get_precision(true_ror, seed_list, true_num, labels, seed_img)
    # detect_rock = get_detect_img(img, detect_img)

def display_pc(precisions, recalls):
        ## Display result
    
    e = 0.2
    ax1.bar(0, np.amax(precisions), align = "center", yerr = e, ecolor = "black")
    ax1.bar(2, np.amax(recalls),  align = "center", yerr = e, ecolor = "black")
    
    ax2.plot(precisions,recalls,"-o")

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

def get_recall2(ror_true, seed_list, num, labels, seed_img):
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
    
    short_rock = 0
    middle_rock = 0
    long_rock = 0
    short = 0
    middle = 0
    longg = 0
    short_range = int(ror_true.shape[0]/3)

    detect_img = non.copy()

    for i in range(1,num):
        
        ## 領域のみに処理する
        reg[labels==i] = 255 # 岩が255
        seed[labels!=i] = 0  # 岩以外の種が0
        
        ## それぞれの領域の重心を求める
        center = np.sum(np.nonzero(reg)[1]) / len(np.nonzero(reg)[1])

        if center < short_range:
            long_rock += 1

        if short_range <= center <= 2 * short_range:
            middle_rock += 1

        # 画像の手前
        if 2 * short_range < center < 3 * long_range:
            short_rock += 1




        ## 検出できてるとき
        if np.array(np.nonzero(seed)).shape[1] != 0:
            detect += 1
            detect_img[labels==i] = 255
            # rock_num.append(i)

            if center < short_range: # 奥
                longg += 1

            if short_range <= center <= middle_range:
                middle += 1

            if middle_range < center < long_range: # 手前
                short += 1


        ## 検出できてないとき
        elif np.array(np.nonzero(seed)).shape[1] == 0:
            nondetect += 1
            detect_img[labels==i] = 150

        ## 初期化と描画
        reg[reg!=0] = 0 

        seed = seed_img.copy()

    recall = 1. * detect/num
    short_recall = 1. * short / short_num
    middle_recall = 1. * middle / middle_num
    long_recall = 1. * longg / long_num

    # cv2.imshow('detect',detect_img)
    # print 'total = {}, detected = {}, NOTdetected = {}'.format(num-1,detect,nondetect)
    # print 'recall = {}'.format(recall)

    return recall, detect, detect_img

def get_precision2(ror_true, seed_list, num, labels, seed_img):
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

def get_detect_img(img,detect_img):
    '''
    '''
    b,g,r = cv2.split((img * 0.8).astype(np.uint8))

    r[detect_img==255]=255
    g[detect_img==150]=255

    result = cv2.merge((b,g,r))

    return result

def range_evaluate(ror):
    '''
        距離に対する評価を行う
        それぞれの領域の重心を求めて
    '''



if __name__ == '__main__':

    ## Ground-Truth data acquisition
    img       = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    label_img = cv2.imread('../../../data/g-t_data/label/spirit118-1.png')
    
    ## Extract features of GT-data
    true_ror, labels, true_num, rock_region, soil_region = label(img,label_img)
    
    ## Predicted data acquisition
    
    
    thresh = range(130,211,10)
    sigmas = range(2,7)
    biases = np.arange(0,1,0.1)
    directions = [np.pi * 3/4,np.pi,np.pi*5/4]

    precisions = list(np.zeros_like(thresh))
    recalls = list(np.zeros_like(thresh))

    total_start = time.time()
    total_attempt = len(thresh)*len(biases)
    
    # もし設定しなかったらこれが読み込まれる
    sigma = 4
    bias = 0.2
    j = 0
    direction = 5 * np.pi/4
    
    # for j, direction in enumerate(directions):
    # for j,bias in enumerate(biases):
    # for j, sigma in enumerate(sigmas):
        # if j>0:
           # plt.hold(True)
    
    for i, th in enumerate(thresh):
            
            start = time.time()
            seed_img = small_rock_detection.main(img, direction=np.pi, thresh=th, sigma = sigma, bias = bias)
            ptime = time.time() - start
            
            seed_list = img2list(seed_img)
            
            if len(seed_list) == 0:
                continue

            ## Template Matching analysis
            recall, pred_num, detect_img = get_recall(true_ror, seed_list, true_num, labels, seed_img)
            precision =     get_precision(true_ror, seed_list, true_num, labels, seed_img)
            detect_rock = get_detect_img(img, detect_img)
            
            ## Display results
            # print '======================='
            # print 'Seed threshold : {}'.format(th)
            # print 'Sun direction  : {}'.format(direction)
            # print 'Gabor sigma    : {}'.format(sigma)
            # print 'Gabor bias     : {}'.format('None')
            print 'Seed quantity  : {}'.format(len(seed_list))
            print 'TRUE Rock      : {}'.format(true_num)
            print 'DETECTED Rock  : {}'.format(pred_num)
            # print 'Precision      : {}'.format(round(precision, 3))
            # print 'Recall         : {}'.format(round(recall   , 3))
            # print 'Process time   : {}'.format(round(ptime, 3))
            # print 'Ateempt number : {}'.format(j*len(biases)+i, len(thresh) * len(biases) )
            # print '======================='

            precisions[i] = precision
            recalls[i] = recall
            cv2.imshow('result',detect_rock)
            cv2.waitKey(-1)
        # display_pc(precisions, recalls)
    ax2.plot(precisions,recalls,"-o",label="Direction = {}".format(direction))
    ax2.set_title('Detection performance by threshold and direction')
    ax2.legend() 


    print 'Total processing time : {}'.format(round(time.time()-total_start, 3))
    plt.show()
    cv2.waitKey(-1)