# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

# 組み込み
import os
import sys
import csv

# ライブラリ
import cv2
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as distances
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_boston

# 自分のレポジトリから


class AnomalyDetection():

    
    def __init__(self):
        
        self.value = 0

    def module1(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        # 図の準備
        nrow = 2
        ncol = 2
        plt.subplots(nrow, ncol, figsize=(16,7))
        gs = gridspec.GridSpec(nrow,ncol)
        axs = [plt.subplot(gs[i]) for i in range(4) ]

        # 一つ目の図の描画 
        rd.seed(0)
        n = 141
        x = np.linspace(0,140,n) # float64でつくる
        y = rd.exponential(1.5, n) * 300 # 乱数じゃない
        col = ["#2F79B0" for _ in range(n)] # colorの生成
        for i in range(5):    
            y[60+i] = rd.exponential(1.5, 1) * 300 + 2000
            col[60+i] = "r"
        axs[0].scatter(x,y, c=col) #散布図の描画
        axs[0].set_xlim(-5,145)
        axs[0].set_xlabel('time',size=12)

        plt.pause(3)
        plt.show()

    def module2(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        ROW = 6
        COLUMN = 10

        row = []
        column = []
        ave = [0.0 for i in range(ROW)]
        vcm = np.zeros((COLUMN, ROW, ROW))
        diff = np.zeros((1, ROW))
        mahal = np.zeros(COLUMN)
        tmp = np.zeros(ROW)

        sam = np.array([[0, 6, 7, 2, 3, 3, 1, 0, 0, 1],
                        [1, 1,11, 6, 0, 2, 1, 4, 1, 2],
                        [2,12,32, 5, 0, 1, 3, 4, 1, 1],
                        [3, 3, 7, 3, 2, 2, 2, 1, 2, 5],
                        [4, 6, 6, 3, 5, 1, 1, 1, 1, 3],
                        [5, 7, 9, 5, 0, 2, 2, 1, 1, 2]])
        sam = sam.astype(np.float64)
        for i in range(len(sam[:,0])):
            ave[i] = np.average(sam[i])
        print ave

    def module3(self):

        # vectorの定義
        vector1 = np.array([1,1]).astype(np.float64)
        vector2 = np.array([2,3]).astype(np.float64)        

        con = np.vstack((vector1, vector2))

        # 逆行列の計算    # viは共分散行列
        vi = np.linalg.inv( con.T )
        print vi

        vector_norm = distances.mahalanobis( vector1, vector2, vi )
        print vector_norm

    def module4(self):
        '''
            入力された一次元配列からanomaly detectionを用いて外れ値を検出する
        '''

        # get data
        img = cv2.imread('../saliency_detection/image/pearl.png')
        b,g,r = cv2.split(img) 
        B,G,R = map(lambda x,y,z: x*1. - (y*1. + z*1.)/2., [b,g,r],[r,r,g],[g,b,b])

        Y = (r*1. + g*1.)/2. - np.abs(r*1. - g*1.)/2. - b*1.
        # 負の部分は0にする
        R[R<0] = 0
        G[G<0] = 0
        B[B<0] = 0
        Y[Y<0] = 0
        rg = cv2.absdiff(R,G)
        by = cv2.absdiff(B,Y)
        img1 = rg
        img2 = by

        rg, by = map(lambda x:x.reshape((len(b[0])*len(b[:,0]),1)),[rg,by])
        data = np.hstack((rg,by))
        data = data.astype(np.float64)
        data = np.delete(data, range( 0,len(data[:,0]),2),0)

        # grid
        xx1, yy1 = np.meshgrid(np.linspace(-10, 300, 500), np.linspace(-10, 300, 500))
        
        # 学習して境界を求める # contamination大きくすると円は小さく
        clf = EllipticEnvelope(support_fraction=1, contamination=0.01)
        print 'data.shape =>',data.shape
        print 'learning...'
        clf.fit(data) #学習 # 0があるとだめっぽいかも
        print 'complete learning!'

        # 学習した分類器に基づいてデータを分類して楕円を描画
        z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        z1 = z1.reshape(xx1.shape)
        plt.contour(xx1,yy1,z1,levels=[0],linewidths=2,colors='r')

        # plot
        plt.scatter(data[:,0],data[:,1],color= 'black')
        plt.title("Outlier detection")
        plt.xlim((xx1.min(), xx1.max()))
        plt.ylim((yy1.min(), yy1.max()))
        plt.pause(.001)
        # plt.show()
        
        cv2.imshow('rg',img1/np.amax(img1))
        cv2.imshow('by',img2/np.amax(img2))



################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    ad = AnomalyDetection()

    # img = cv2.imread(sys.argv[1])
    # if img is None:
        # print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        # sys.exit()
    ad.module4()

    

    # cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    