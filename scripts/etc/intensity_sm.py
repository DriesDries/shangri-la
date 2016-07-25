# -*- coding: utf-8 -*-
'''
    IntensitySMの実装
    Usage: $ python intensity_sm.py <argv> or python intensity_sm.py
    argv : image,3ch
    dst  : Saliency Map, 1ch
''' 

import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import math
from pylab import *
import matplotlib.tri as mtri
from matplotlib import cm 
from matplotlib.mlab import griddata

class SaliencyMap():

    def __init__(self):
        self.sample = 0


    def SaliencyMap(self,img):
        '''
            入力された画像のSaliency Mapを求める
            args : img  -> 入力画像
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''
        # 前処理
        # img = self.PostProcessing(img)

        # Saliency Mapの入手
        IntensityCM = self.GetIntensityCM(img)

        cv2.imshow('img',img)


        return 0

    def GetIntensityCM(self,img):
        '''
           入力画像のIntensity Couspicuity Mapを求める
           argv: img         -> 入力画像
           dst : IntensityCM -> 強度の顕著性マップ
        '''
        # Get intensity image
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Generate Gaussian pyramid　[0]元画像、全部で10枚
        intensity_pyr = self.GetGaussianPyramid(img) 
        intensity_pyr = np.array(intensity_pyr).astype(np.float)

        # Get 6 Difference of Gaussians, float64へ変換
        intensity_dog = self.GetDoG(intensity_pyr) 

        # Get saliency map
        intensity_sm = self.GetSaliencyMap(intensity_dog)
        cv2.imshow('saliency map',intensity_sm/np.amax(intensity_sm))
        # self.test(intensit_pyr)

        return 0


    def GetGaussianPyramid(self,img):
        '''
            入力された画像のgaussian pyramidを計算する
            args : img    -> Mat,Uint8,1chの画像
            dst  : pyr  -> 元画像合わせて10階層のgaussian画像が入ったlist，それぞれargsと同じサイズ
                           pyr[i].shapeは，src.shape / 2**i
        '''
        # 準備
        pyr  = range(10)
        pyr[0]  = img 

        # Create Gaussian pyramid
        for i in range(1,10):
            pyr[i] = cv2.pyrDown(pyr[i-1])
        
        # Resize pyramid 
        for i in range(10):
            pyr[i] = cv2.resize(pyr[i],(len(img),len(img)), interpolation = cv2.INTER_LINEAR)
            # cv2.imshow('pyr%s'%i,pyr[i])
        return pyr

    def GetDoG(self,pyr):
        '''
           入力されたGauPyrに対してDoGを計算する
           階層が3つ離れているものと4つ離れているものの差分をとる
            args : pyr     ->   各階層のgaussian画像が入ったlist
            dst  : DoG     ->   特定の階層のDoGが入ったlist

        '''
        # sはscale
        DoG = range(6)
        for i,s in enumerate(range(2,5)):
            DoG[2*i]  = cv2.absdiff(pyr[s],pyr[s+2])
            DoG[2*i+1] = cv2.absdiff(pyr[s],pyr[s+3])
        return DoG
      
    def GetSaliencyMap(self,dog):
        '''
            DoGからSMを形成する
        '''

        sm = np.zeros_like(dog[0])

        for i in range(6):
            cv2.imshow('dog%s'%i,dog[i]/np.amax(dog[i]))
            sm = sm + dog[i]

        return sm

    def GetIntensityImg(self,img):
        '''
            入力画像のIntensityImgを返す
            argv: img ->  3chのColorImg
            dst : IntensityImg -> 1chのImg
        '''
        b,g,r = cv2.split(img)
        IntensityImg = b/3. + g/3. + r/3.
        print 'rgb',np.amax(IntensityImg)
        R,G,B,Y = self.GetRGBY(img) # float #ok
        # IntensityImg = r
        print 'r',np.amax(IntensityImg)
        
        ### display results #####################################
        # cv2.imshow('intensity',IntensityImg/np.amax(IntensityImg))
        # cv2.imshow('r',R/np.amax(R))
        # cv2.imshow('g',G/np.amax(G))
        # cv2.imshow('b',B/np.amax(B))
        #########################################################

        return IntensityImg
    def PostProcessing(self,img):
        '''
            前処理、フィルタ処理とか
            args : img  -> 入力画像
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''

        out = cv2.bilateralFilter(img,10,50,50)

        return out
################################################################################
# main
################################################################################
if __name__ == '__main__':

    # class recognition
    SM = SaliencyMap()


    ### get image #################
    # src = sys.argv[1]
    # img = cv2.imread(src)
    # img = cv2.imread('../../image/rock/sol729.jpg')
    # img = cv2.imread('../../image/rock/spiritsol118navcam.jpg',0)
    img = cv2.resize(cv2.imread('../../image/rock/11.png'),(512,512))
    
    # img = cv2.resize(img,(512,512))
    if img is None:
        print  '    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'\
              +'    !!!!!!There is not %s'%src + '!!!!!!!!\n'\
              +'    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        sys.exit()
    # img = cv2.resize(img,(512,512))
    # img[np.sum(img,axis=2)==0] = 255
    ################################

    sm = SM.SaliencyMap(img)


    
    cv2.waitKey(-1)