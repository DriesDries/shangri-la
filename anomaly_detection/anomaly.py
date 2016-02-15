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
from matplotlib import gridspec
from matplotlib import animation as ani
import pandas as pd


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

        # plt.pause(3)
        plt.show()



    def module2(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        out = img
        return out


################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    ad = AnomalyDetection()

    # img = cv2.imread(sys.argv[1])
    # if img is None:
        # print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        # sys.exit()
    # ad.module1(0)

    sam = pd.read_csv('tests.csv')
    print sam

    # cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    