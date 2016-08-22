# -*- coding: utf-8 -*-
'''
    Image clustering based on Texton Map
    
    Usage: $ python texture_map.py

    ToDo:
        - rockとsoilに分けていろんな特徴抽出しまくる
            - 色
            - カイ二乗距離
            - 
''' 

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import cluster
from scipy import stats
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

def main(img):
    '''
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    
    # generate filter bank
    filter_bank = get_filter_bank() # float64
    # print filter_bank.shape

    # convolution
    responses = convolution(img, filter_bank)
    print responses.shape

    # for i in range(13):
        # res = cv2.normalize(responses[i], 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow('res',res)
        # cv2.waitKey(-1)

    texton = get_texton2(responses)
    # texton = texton.astype(np.float)
    # print texton.shape
    # cv2.imshow('img',img)
    # cv2.imshow('maxton',texton[0]/np.amax(texton[0]))
    # cv2.imshow('minton',texton[1])

    # clustering
    texton_map = clustering(img, texton)

def get_texton2(responses):
    '''
        texton[0] -> 最も反応したフィルタ
        texton[1,2,3,4] -> 上下左右-そのピクセルのカイ二乗距離
    '''
    width = responses.shape[1]
    row = responses.shape[2]
    
    responses = responses.astype(np.int16)
    
    texton = np.zeros((2,responses.shape[1],responses.shape[2]))

    for y in range(width):
        # print y-1
        for x in range(row):
            if x == 0 or y == 0 or x == width-1 or y == width-1:
                pass
            else:
                texton[0,y,x] = np.argmax(responses[:,y,x])
                # u = responses[:,y-1,x]
                # c = responses[:,y,x]
                # stats.chisquare(u, c)[0]

    return texton

def clustering(img, texton, N=10):
    '''
        それぞれのヒストグラムのtextonに応じてclusteringを行う
    '''
    # clustering
    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # textonの変換,両方ともflatにしてstackする
    t0 = texton[0].flatten()
    # t1 = texton[1].flatten()
    t1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).flatten()    
    t = np.vstack((t0,t1)).T
    km = cluster.KMeans(n_clusters=N)
    pred = km.fit_predict(t)
    print 'pred',pred.shape

    # clusteringの結果
    t = np.array(t)
    t_random = np.vstack((t.T,pred)).T
    t_random = random.sample(t_random,1000)
    t_random = np.array(t_random)
    print t.shape,t_random.shape
    # print t_random[:,1]
    plt.scatter(t_random[:,0], t_random[:,1], c=t_random[:,2])
    plt.pause(1)

    # texton mapの生成
    texton_map = np.zeros_like(gimg)
    for y in range(gimg.shape[0]):
        for x in range(gimg.shape[1]):
            texton_map[y,x] = pred[gimg.shape[0]*y + x] # ずれてる？
    texton_map = texton_map.astype(np.uint8)


    red = 0
    green = 255
    b,g,r = cv2.split(img)

    # 結果の表示
    for i in range(N):
        print i
        # b,g,r = cv2.split(img)
        r[texton_map == i] = red
        g[texton_map == i] = 0
        result = cv2.merge((b,g,r))
        cv2.imshow('result',result)
        cv2.waitKey(-1)
        red += 20
        # green -= 20

def get_filter_bank():
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    # 角度の準備
    rad = np.arange(0,181,15).astype(float)
    deg = np.pi * rad / 180
    print 'rad',rad
    bank = []

    for angle in deg:
        kernel = cv2.getGaborKernel(ksize = (5,5), sigma = 5,theta = angle, lambd = 5, gamma = 5,psi = np.pi * 1/2)
        bank.append(kernel)
    bank = np.array(bank)
    return bank

def convolution(img, filter_bank):

    fimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float64)
    responses = []

    for i,kernel in enumerate(filter_bank):
        responses.append(cv2.filter2D(fimg, cv2.CV_64F, kernel))

    responses = np.array(responses)

    return responses

def analysis(img, rock, soil, brock):
    '''
        Intensity -> 分離は難しい
        Var       -> 岩の方が大きい

    '''
    brock = cv2.normalize(brock, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
    bsoil = 255 - brock
    labels, num = sk.label(brock, return_num = True)

    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grock = cv2.cvtColor(rock,cv2.COLOR_BGR2GRAY)
    gsoil = cv2.cvtColor(soil,cv2.COLOR_BGR2GRAY)

    ## 図の用意
    # fig = plt.figure(figsize=(16,9))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.set_title('Rock')
    # ax2.set_title('Soil')

    ## それぞれのflattenの取得
    imgten = gimg.flatten()
    rockten = grock.flatten()
    soilten = gsoil.flatten()
    rockten = rockten[np.nonzero(rockten)]
    soilten = soilten[np.nonzero(soilten)]

    ## Var
    # print rockten.var()
    # print soilten.var()

    ## Intensity histgram
    # ax1.hist(rockten, bins=50, normed=1, color='blue', alpha=0.5)
    # ax2.hist(soilten, bins=50, normed=1, color='blue', alpha=0.5)

    ## Texture analysis
    filter_bank = get_filter_bank()
    textens = []

    # for img in [img,rock,soil]:
        # responses = convolution(img, filter_bank)
        # texton = get_texton(responses) # それぞれに最も反応したフィルタの番号
        # texten = texton.flatten()
        # textens.append(texten)

    ## histgramだと変わらない
    # ax1.hist(textens[0], color='blue', alpha=0.5)
    # ax1.hist(textens[1], color='green', alpha=0.5)
    # ax2.hist(textens[2], color='blue', alpha=0.5)

    ## Responses同士で比較
    # これに岩とsoil同時にプロットして、すべてのフィルタ同士でやってみる

    img_r = convolution(img, filter_bank)
    rock_r = convolution(rock, filter_bank)
    soil_r = convolution(soil, filter_bank)

    dis = 1
    

    for dis in range(1,30):
      if dis >= img_r.shape[0]:
            print 'COMPLETE!' ; break
      for i in range(100):
        # ここで図を切り替える
        if i + dis >= img_r.shape[0]:
            break
        print '{} vs {} , dis = {}'.format(i,i+dis,dis)
        plt.title('{} vs {} , dis = {}'.format(i,i+dis,dis))
        
        for i2,responses in enumerate([rock_r,soil_r]):
            r1 = responses[i]
            r2 = responses[i+1]
            r1 = r1.flatten()
            r2 = r2.flatten()

            r = np.vstack((r1,r2)).T
            r_random = np.array(random.sample(r,1000)).T
            
            if i2 == 0:
                plt.scatter(r_random[0],r_random[1],color='r')
            if i2 == 1:
                plt.scatter(r_random[0],r_random[1],color='b')

        plt.pause(0.2)
        plt.clf()
    ## それぞれの岩ごとに分析、テクスチャの分散を見る、最大値だけでなくて他のも見る, filterbank見る 

def get_texton(responses):
    '''
        responses[i]に各画像のが入ってる
        つまり、responses[:][y,x]に各ピクセルのresponseが入ってる
    '''
    texton = np.zeros((responses.shape[1], responses.shape[2]))

    for y in range(responses.shape[1]):
        for x in range(responses.shape[2]):

            texton[y,x] = np.argmax(responses[:,y,x])
            # texton[y,x,1] = np.argmin(responses[y,x,:])

            if 128 <= np.amax(responses[:,y,x]) <= 132:
                texton[y,x] = 0
                # texton[y,x,1] = 0

    texton = texton.astype(np.uint8)

    return texton


if __name__ == '__main__':

    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    rock = cv2.imread('../../../data/g-t_data/rock_region/spirit118-1.png')
    soil = cv2.imread('../../../data/g-t_data/soil_region/spirit118-1.png')
    label_img= cv2.imread('../../../data/g-t_data/label/spirit118-1.png')

    true_ror, labels, true_num, rock_region, soil_region = label(img,label_img)

    analysis(img,rock,soil, true_ror)







    cv2.waitKey(-1)