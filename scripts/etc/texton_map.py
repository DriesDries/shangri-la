# -*- coding: utf-8 -*-
'''
    Image clustering based on Texton Map
    
    Usage: $ python texture_map.py
    src  : img
    dst  : texton_map

    ToDo:
        
        ✔︎ 正しいtexton map
            ✔︎ 縦シマに現れてるのおかしい気がする
            ✔︎ responseの正規化
        - Iの影響取り除く
            - filteringするときに正規化定数で割る？
        - カーネルの大きさ 
            - radius
        - filterbank変える？
        
        - 距離
            ✔︎ center
            - 距離の扱い方、正規化どうするか
''' 

import os
import sys
import time
import math

import random
import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import cluster
from scipy import stats
import collections

import filterbank2 as fb

N = 8
radiuses = range(2,30,2)
radius = 4

def main(img, radius):
    '''
        args :      -> 
        dst  :      -> 
        param:      -> 

        dis[i,j] -> cluster[i]とcluster[j]のcenter間のユークリッド距離
    '''

    # generate filter bank ; total 38 filters
    edges,bars,rots = makeRFSfilters() 
    schmids         = fb.main(radius*2+1)

    ## Convolution; Normalize 0 - 1
    responses = convolution(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float64), edges, bars, rots, schmids)
    responses = norm(responses)

    ## Generate texton map
    # texton_map, centers = get_textonmap(img,responses)

    ## Calculate distance of each clusters
    # dis = get_distance(texton_map, centers)

    ## texton mapで最も大きいクラスタの数を返す



    # fig = plt.figure(figsize=(16,9))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(111)

    # print np.max(dis),np.min(dis),np.max(texton_map),np.min(texton_map)
    # ax1.imshow(dis, cmap='gray',interpolation='nearest',vmin=0, vmax=1)
    # txm = ax2.imshow(texton_map, cmap='jet')
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # fig.colorbar(txm, cax=cax)
    # plt.pause(-1)

    # for i in range(N):
    #     b,g,r = cv2.split(img)
    #     r[texton_map==i] = 255
    #     res = cv2.merge((b,g,r))
    #     cv2.imshow('res',res)
    #     cv2.waitKey(-1)

    texton_map = 0
    dis = 0
    return texton_map, dis, np.array(responses)

def get_distance(texton_map, centers):
    '''
        clusterの中心間の距離を測定する
        dis[i,j] -> distance between c[i] and c[j]
    '''

    dis = np.zeros((N,N))
    row = 0

    for j,c1 in enumerate(centers):
        for i,c2 in enumerate(centers):
            dis[j,i] = np.linalg.norm(c1-c2)   # ユークリッド距離
            # dis = stats.chisquare(c1, c2)[0] # カイ二乗距離

    dis = dis/np.max(dis)

    return dis

def convolution(img, edges, bars, rots, schmids):
    '''
        38種類のfilterで畳み込みを行い、
        いくつかの種類ごとに最大の反応を示した8個のresponseを返す
        filter -> float64

        edges -> gabor filter  同じスケールの違う角度が6しゅるい　x 3つのスケール

    '''
    responses = []
    sums = []
    max_responses = []

    # gabor filterから3つ
    for i,kernel in enumerate(edges):

        response = cv2.filter2D(img, cv2.CV_64F, kernel)
        responses.append(cv2.filter2D(img, cv2.CV_64F, kernel))
        sums.append(response.sum())
            
        if (i+1)%6 == 0 :
            # 最大値がふくまれているresを保存する？
            # print np.argmax(sums),np.max(sums)
            max_responses.append(responses[np.argmax(sums)])

            # 初期化
            responses = []
            sums = []

    for i,kernel in enumerate(bars):
        response = cv2.filter2D(img, cv2.CV_64F, kernel)
        responses.append(cv2.filter2D(img, cv2.CV_64F, kernel))
        sums.append(response.sum())
            
        if (i+1)%6 == 0 :
            # 最大値がふくまれているresを保存する？
            # print np.argmax(sums),np.max(sums)
            max_responses.append(responses[np.argmax(sums)])

            # 初期化
            responses = []
            sums = []
    

    for i,kernel in enumerate(rots):
        max_responses.append(cv2.filter2D(img, cv2.CV_64F, kernel))
    
    ## If use Schmid filter bank
    # max_responses = []
    # for i,kernel in enumerate(schmids):
        # max_responses.append(cv2.filter2D(img, cv2.CV_64F, kernel))



    ## 結果の表示
    # for i in range(8):
        # plt.imshow(max_responses[i])
        # plt.pause(1)

    return max_responses

def get_textonmap(img, responses):
    '''
        k-meansでresponsesをクラスタリングする
    '''
    
    ## 配列の作成
    arr = []
    for res in responses:
        arr.append(res.flatten())
    arr = np.array(arr).T

    ## サンプリングする場合
    # arr = np.array(random.sample(arr,1000))
    # arr = arr[:,0:8] # test function

    ## arrをクラスタリングする
    kmean = cluster.KMeans(n_clusters=N, init='k-means++', n_init=10, max_iter=300,tol=0.0001,precompute_distances='auto', verbose=0,random_state=None, copy_x=True, n_jobs=1)
    arr_cls = kmean.fit(arr)
    pred_labels  = arr_cls.labels_
    pred_centers = arr_cls.cluster_centers_
    
    # texton mapの生成
    size = int(math.sqrt(len(pred_labels)))
    texton_map = np.reshape(pred_labels, (size,size))



    return texton_map, pred_centers

def norm(imgs):
    '''
        max_responsesを正規化する
        入力画像を正規化して返す
    '''
    imgs += abs(np.min(imgs))
    imgs = 10. * imgs / np.max(imgs)

    # for i,img in enumerate(imgs):
        # cv2.imshow('img',img)        
        # cv2.waitKey(-1)

    return imgs

def makeRFSfilters(sigmas=[1, 2, 4], n_orientations=6):
    """ Generates filters for RFS filterbank.
    Parameters
    ----------
    radius : int, default 28
        radius of all filters. Size will be 2 * radius + 1
    sigmas : list of floats, default [1, 2, 4]
        define scales on which the filters will be computed
    n_orientations : int
        number of fractions the half-angle will be divided in
    Returns
    -------
    edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains edge filters on different scales and orientations
    bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains bar filters on different scales and orientations
    rot : ndarray (2, 2*radius+1, 2*radius+1)
        contains two rotation invariant filters, Gaussian and Laplacian of
        Gaussian
    """
    def make_gaussian_filter(x, sigma, order=0):
        if order > 2:
            raise ValueError("Only orders up to 2 are supported")

        # compute unnormalized Gaussian response
        response = np.exp(-x ** 2 / (2. * sigma ** 2))
        if order == 1:
            response = -response * x
        elif order == 2:
            response = response * (x ** 2 - sigma ** 2)

        # normalize
        response /= np.abs(response).sum()
        return response

    def makefilter(scale, phasey, pts, sup):
        gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
        gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
        f = (gx * gy).reshape(sup, sup)
        
        # normalize
        f /= np.abs(f).sum() # L1 norm

        return f

    support = 2 * radius + 1
    x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
    orgpts = np.vstack([x.ravel(), y.ravel()])

    rot, edge, bar = [], [], [] # 空の用意
    for sigma in sigmas:
        for orient in xrange(n_orientations):
            # Not 2pi as filters have symmetry
            angle = np.pi * orient / n_orientations
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
            edge.append(makefilter(sigma, 1, rotpts, support))
            bar.append(makefilter(sigma, 2, rotpts, support))
    length = np.sqrt(x ** 2 + y ** 2)
    rot.append(make_gaussian_filter(length, sigma=10)) # gaussian filter
    rot.append(make_gaussian_filter(length, sigma=10, order=2)) # log filter



    return edge, bar, rot

if __name__ == '__main__':


    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    # img = cv2.imread('../../../data/test_image/sample/image14.png')
    # img = cv2.imread('../../../data/test_image/sample/texture.jpg')
    # rock = cv2.imread('../../../data/g-t_data/rock_region/spirit118-1.png')
    # soil = cv2.imread('../../../data/g-t_data/soil_region/spirit118-1.png')
    img = cv2.resize(img, (500,500))

    ## Main processing
    for radius in radiuses:
        print 'radius = {}'.format(radius)
        texton_map, dis, responses = main(img.copy(), radius)

        # Draw result
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax1.imshow(dis, cmap='gray',interpolation='nearest',vmin=0, vmax=1)
        # src = ax1.imshow(img, cmap=plt.get_cmap('gray'))
        tm  = ax2.imshow(texton_map, cmap=plt.get_cmap('jet'))
        fig.colorbar(tm, cax=cax)
        # plt.savefig("../../fig/radius{}.png".format(radius))
        plt.show()


    # ax3 = fig.add_subplot(121)
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # d = ax3.imshow(dis, interpolation='nearest',cmap=plt.get_cmap('jet'))
    # fig.colorbar(d, cax=cax)

    ##draw distance
    # ax4 = fig.add_subplot(111)
    # divider = make_axes_locatable(ax4)
    # dis_map = np.zeros_like(texton_map).astype(np.float64)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # for j in range(N):
    #     for i in range(N):
            
    #         if i == j:
    #             dis_map[texton_map==i] = 0
            
    #         else:
    #             dis_map[texton_map==i] = dis[i,j]
        
    #     di = ax4.imshow(dis_map, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    #     fig.colorbar(di, cax=cax)
    #     plt.pause(2)

        # cv2.imshow('di',dis_map)
        # cv2.waitKey(-1)

    ## クラスタごとにマスク
    for i in range(N):
        b,g,r = cv2.split(img)
        r[texton_map==i]= 255
        res = cv2.merge((b,g,r))
        cv2.imshow('res',res)
        cv2.waitKey(-1)
    plt.show()
    cv2.waitKey(-1)