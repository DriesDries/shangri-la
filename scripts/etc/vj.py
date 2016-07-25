# -*- coding: utf-8 -*-
'''
Rock Detection based on Saliency Map and Region Growing Algolithm
Usage: $ python rock_detection.py
argv : Image,3ch
dst  : Region of Rocks, 1ch
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.measure as sk
import skimage.segmentation as skseg
import scipy.ndimage as nd

from scipy import misc
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import time
import math
import copy

from get_maxima import getMaxima as getmaxima


class SaliencyMap():

    def __init__(self):
        # いくつのDoGを足し合わせるか
        self.scale = 6
        
        # gabor kernelのパラメータ
        self.ksize  = (5,5)
        self.sigma = 5
        self.lambd = 5
        self.gamma = 1
        self.GaborKernel_0   = cv2.getGaborKernel(ksize = self.ksize, sigma = self.sigma,theta = 0, lambd = self.lambd, gamma = self.gamma)
        self.GaborKernel_45  = cv2.getGaborKernel(self.ksize, self.sigma, 45, self.lambd,  self.gamma)
        self.GaborKernel_90  = cv2.getGaborKernel(self.ksize, self.sigma, 90, self.lambd,  self.gamma)
        self.GaborKernel_135  = cv2.getGaborKernel(self.ksize, self.sigma, 135, self.lambd,  self.gamma)  
    
    def SaliencyMap2(self,img):
        '''
            入力された画像のgaussian pyramidをつくり、その中の任意のスケールで差分を取り、
            それを返す
        '''
        intensity_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Create Gaussian pyramid　[0]元画像、全部で10枚
        pyr = self.GetGaussianPyramid(intensity_img) 
        pyr = np.array(pyr).astype(np.float)

        dog = cv2.absdiff(pyr[2],pyr[5])
        # dog2 = cv2.absdiff(pyr[1],pyr[4])
        # dog3 = dog + dog2

        # dog = cv2.dilate(dog,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
        # dog = cv2.erode(dog,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))

        dog = cv2.normalize(dog, 0, 255, norm_type = cv2.NORM_MINMAX)
        dog = dog.astype(np.uint8)

        return dog


    def SaliencyMap(self,img):
        '''
            入力された画像のSaliency Mapを求める
            args : img  -> 入力画像,3ch
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''
        # 前処理
        # img = self.PostProcessing(img)
        # cv2.imshow('post',img)

        # Saliency Mapの入手
        IntensityCM = self.GetIntensityCM(img)

        return IntensityCM

    def PostProcessing(self,img):
        '''
            前処理、フィルタ処理とか
            args : img  -> 入力画像
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''

        out = cv2.bilateralFilter(img,10,50,50)

        return out

    def GetIntensityCM(self,img):
        '''
           入力画像のIntensity Couspicuity Mapを求める
           argv: img         -> 入力画像
           dst : IntensityCM -> 強度の顕著性マップ
        '''
        # Get intensity image
        intensity_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Create Gaussian pyramid　[0]元画像、全部で10枚
        intensity_pyr = self.GetGaussianPyramid(intensity_img) 
        intensity_pyr = np.array(intensity_pyr).astype(np.float)

        # Get 6 Difference of Gaussians, float64へ変換
        intensity_dog = self.GetDoG(intensity_pyr) 
        intensity_dog = np.array(intensity_dog).astype(np.float)

        # Get saliency map
        intensity_sm = self.GetSaliencyMap(intensity_dog)
        sm = (intensity_sm/np.amax(intensity_sm) * 255).astype(np.uint8)

        return sm

    def GetTextureCM(self,img):

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Get each Orientation images pyramid
        Orientation0Pyr, Orientation45Pyr, Orientation90Pyr, Orientation135Pyr = self.GetOrientationPyr(img)
        
        # # Create feature maps
        Orientation0FM = self.GetDoG(Orientation0Pyr)
        Orientation45FM = self.GetDoG(Orientation45Pyr)
        Orientation90FM = self.GetDoG(Orientation90Pyr)
        Orientation135FM = self.GetDoG(Orientation135Pyr)

        Orientation0FM = np.array(Orientation0FM).astype(np.float)
        Orientation45FM = np.array(Orientation45FM).astype(np.float)
        Orientation90FM = np.array(Orientation90FM).astype(np.float)
        Orientation135FM = np.array(Orientation135FM).astype(np.float)

        # # create Couspicuity Map floatなのに注意！
        Orientation0CM = self.GetCM(Orientation0FM)
        Orientation45CM = self.GetCM(Orientation45FM)
        Orientation90CM = self.GetCM(Orientation90FM)
        Orientation135CM = self.GetCM(Orientation135FM)        
        OrientationCM     = Orientation0CM + Orientation45CM + Orientation90CM + Orientation135CM
        # OrientationCM = OrientationCM.astype(np.uint8)

        OrientationCM = cv2.normalize(OrientationCM, 0, 255, norm_type = cv2.NORM_MINMAX)
        OrientationCM = OrientationCM.astype(np.uint8)
        cm = OrientationCM
        cm[cm<50]=0
        cv2.imshow('ori',cm)

        return cm


    def GetOrientationPyr(self,img):
        '''
            入力画像からorientationのfeature mapを生成して返す
            argv: img -> 入力画像，1ch,float
            dst : rg,by -> 正規化されたrgとbyの画像，!!!float!!!,1ch
        '''
        GauPyrImg = self.GetGaussianPyramid(img) 

        gabor0  = range(10)
        gabor45 = range(10)
        gabor90 = range(10)
        gabor135= range(10)        
        
        # それぞれgrayのgaupyrをgabor filterにかけたもの # uint8
        for i in range(10):
            gabor0[i]   = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_0)
            gabor45[i]  = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_45)
            gabor90[i]  = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_90)
            gabor135[i] = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_135)
            # gabor180[i] = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_180)
            # cv2.imshow('gabor%s'%i,gabor0[i])
        return gabor0, gabor45, gabor90, gabor135


    def GetIntensityImg(self,img):
        '''
            入力画像のIntensityImgを返す
            argv: img ->  3chのColorImg
            dst : IntensityImg -> 1chのImg
        '''
        b,g,r = cv2.split(img)
        IntensityImg = b/3. + g/3. + r/3.
        print 'rgb',np.amax(IntensityImg)
        R,G,B,Y = self.GetRGBY(img) # float
        # IntensityImg = r
        print 'r',np.amax(IntensityImg)

        return IntensityImg

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
            pyr[i] = cv2.resize(pyr[i],(512,512), interpolation = cv2.INTER_LINEAR)
        
        return pyr

    def GetDoG(self,pyr):
        '''
           入力されたGauPyrに対してDoGを計算する
           階層が3つ離れているものと4つ離れているものの差分をとる
            args : pyr     ->   各階層のgaussian画像が入ったlist
            dst  : DoG     ->   特定の階層のDoGが入ったlist

        '''
        # sはscale
        FM = range(6)
        for i,s in enumerate(range(2,5)):
            FM[2*i]  = cv2.absdiff(pyr[s],pyr[s+3])
            FM[2*i+1] = cv2.absdiff(pyr[s],pyr[s+4])

        return FM
      
    def GetCM(self,dog):
        '''
            DoGからSMを形成する
        '''

        sm = np.zeros_like(dog[0])

        for i in range(self.scale):
            sm = sm + dog[i]
            # cv2.imshow('dog%s'%i,dog[i]/np.amax(dog[i]))

        return sm

class DisplayResult():

    def display_histgram(self,img):
        '''
            処理の概要
            args :      -> 1ch or 3ch
            dst  :      -> 
            param:      -> 
        '''
        hist=img.ravel()
        plt.hist(hist,256,[0,256])
        plt.xlim([0,256])
        plt.pause(0.01)

    def display_3D(self,img):
        '''
            入力画像を3Dで表示する
            args: 1ch image
        '''
        # データの準備
        x = np.arange(0, len(img[0]), 1)
        y = np.arange(0, len(img[1]), 1)
        X, Y = np.meshgrid(x, y) 
        Z = img

        # plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)

        # 設定
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_zlim(0, 300)
        ax.set_title('Image')
        ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap = 'jet',linewidth=0)
        # ax.plot_wireframe(X,Y,Z, cmap = 'Greys', rstride=10, cstride=10)

        plt.pause(.001) # これだけでok
        # plt.show()

################################################################################
# main
################################################################################
if __name__ == '__main__':

    # read class
    sm = SaliencyMap()

    # get image
    img = cv2.resize(cv2.imread('../../image/rock/spiritsol118navcam.jpg',0),(512,512))
    # img = cv2.resize(cv2.imread('../../image/rock/sol729.jpg',0),(512,512))
    # img = cv2.resize(cv2.imread('../../image/rock/surface.png',0),(512,512))
    cv2.imshow('src',img)

    # main process
    # rd.rock_detection(img)

    kernel   = cv2.getGaborKernel(ksize = (5,5), sigma = 5,theta = np.pi, lambd = 5, gamma = 5,psi = np.pi * 1/2)


    kernel = np.array(kernel)
    # kernel = cv2.normalize(kernel, 0, 255, norm_type = cv2.NORM_MINMAX)
    # kernel = kernel.astype(np.uint8)
    # cv2.imshow('kernel',kernel)
    print kernel


    print np.amax(kernel),np.amax(img),kernel.dtype,img.dtype
    
    gabor  = cv2.filter2D(img, cv2.CV_32F, kernel)
    gabor = cv2.normalize(gabor, 0, 255, norm_type = cv2.NORM_MINMAX)
    gabor = gabor.astype(np.uint8)
    gabor[gabor<120] = 0
    print np.amax(gabor),np.amin(gabor)
    cv2.imshow('ga',gabor)

    




    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # src = copy.deepcopy(img)
    # for i in range(100,150,1):
    #     print i
    #     b,g,r = cv2.split(src)
    #     r[gabor>i] = 255
    #     gabor[gabor<i] = 0
    #     img = cv2.merge((b,g,r))
    #     cv2.imshow('img',img)
    #     # cv2.imshow('gabor',gabor)
    #     cv2.waitKey(1000)

    cv2.waitKey(-1)