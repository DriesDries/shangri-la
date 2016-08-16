# -*- coding: utf-8 -*-
'''
    Image clustering based on Texton Map
    
    Usage: $ python texture_map.py

    ToDo:
        - リストっぽくしなきゃクラスタリングできない
            - こうなると画像上の位置考慮できなくなる
        - 一番相関が小さいのは位相がπ変わったとき
        - filterしたとき位相がpiずれると符合が変わっただけになる
        - おそらく結果おかしい
        - heatmapで表示
''' 
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import cluster
from scipy import stats

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
    rad = np.arange(0,181,30).astype(float)
    deg = np.pi * rad / 180

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

def get_texton(responses):
    '''
        responses[i]に各画像のが入ってる
        つまり、responses[:][y,x]に各ピクセルのresponseが入ってる
    '''

    texton = np.zeros((2,responses.shape[1],responses.shape[2]))

    for y in range(responses.shape[1]):
        for x in range(responses.shape[2]):

            texton[0,y,x] = np.argmax(responses[:,y,x])
            texton[1,y,x] = np.argmin(responses[:,y,x])
            if 128 <= np.amax(responses[:,y,x]) <= 132:
                texton[0,y,x] = 0
                texton[1,y,x] = 0


    texton = texton.astype(np.uint8)

    return texton


if __name__ == '__main__':

    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    # img = cv2.resize(img,(100,100))

    # img = cv2.imread('../../../image/sample/image3.png')
    # gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print gimg.shape
    # print img.shape,img.shape[0],img.shape[1],img[0].shape

    main(img)

    cv2.waitKey(-1)
    