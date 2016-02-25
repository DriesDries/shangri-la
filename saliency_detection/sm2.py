# -*- coding: utf-8 -*-
'''
    A Model of Saliency-Based Visual Attention for Rapid Scene Analysisの実装
    入力画像のSaliencyMapを求め，出力する
    Usage: $ python saliency_map.py <argv> 
    argv : image,3ch
    dst  : Saliency Map, 1ch
    imshowはmainとclassでのメインの関数のみで行う
    それ以外でしたいときはreturnで渡すもしくはdisplay resultsで
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



class ConvolutionMatrix():

    def __init__(self):

        self.rows = 50
        self.cols = 3

    def ViewKernel(self,kernel):
        '''
            入力されたkernelをヒートマップで表示する
        '''
        # データの準備
        x = arange(0, 99, 1)
        y = arange(0, 99, 1)
        X, Y = meshgrid(x, y) 
        Z = kernel

        # define grid. # griddata使う
        # xi = np.linspace(-2.1, 2.1, 100)
        # yi = np.linspace(-2.1, 2.1, 200)
        # grid the data.
        # zi = griddata(x, y, Z, X, Y, interp='linear')        
        # print np.array(zi).shape
        # グラフの設定
        '''2D'''
        # plt.xlabel('pixel')
        # plt.ylabel('pixel')
        # plt.title('Kernel Size = %s,'%self.k+'  Sigma = %s,'%self.sigma+'  Theta = %s,'%0+'  Lambda = %s,'%self.lambd+'  Gamma = %s'%self.gamma)
        # plt.pcolor(X, Y, Z)
        # plt.colorbar()

        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # classよんでる？
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_title('Convolution Matrix')

        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',)
        ax.plot_wireframe(X,Y,Z, cmap = cm.RdPu, rstride=2, cstride=2)

        plt.imshow(Z)
        # plt.draw(-1)

        plt.pause(-1)   


    def GetXkernel(self,rows):
        '''
            x方向の微分kernel
        '''
        kernel = [-1,0,1]
        _ = kernel
        for i in range(self.rows - 1):
            kernel = np.vstack((kernel,_))
        
        return kernel        

    def GetGaborKernel(self,rows,cols,angle):
        '''
        入力された
        '''
        pass



class SaliencyMap():

    def __init__(self):
        
        '''Gabor Filer''' 
        self.k = 99 # odd
        self.ksize  = (self.k,self.k)
        self.sigma = 20
        self.lambd = 5
        self.gamma = 1
        self.GaborKernel_0   = cv2.getGaborKernel(ksize = self.ksize, sigma = self.sigma,theta = 0, lambd = self.lambd, gamma = self.gamma)
        self.GaborKernel_45  = cv2.getGaborKernel(self.ksize, self.sigma, 45, self.lambd,  self.gamma)
        self.GaborKernel_90  = cv2.getGaborKernel(self.ksize, self.sigma, 90, self.lambd,  self.gamma)
        self.GaborKernel_135  = cv2.getGaborKernel(self.ksize, self.sigma, 135, self.lambd,  self.gamma)       

    def saliency_map(self,img):
        '''
            入力された画像のSaliency Mapを求める
            args : img  -> 入力画像
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''
        # Get Conspicuity Map of each features
        ColorCM = self.GetColorCM(img)
        OrientationCM = self.GetOrientationCM(img)
        IntensityCM = self.GetIntensityCM(img)
        
        # create Saliency Map
        SM = IntensityCM/3. + ColorCM/2. + OrientationCM/6.

        # 最大値求める
        maxCMval = max( np.amax(SM),max( max(np.amax(IntensityCM), np.amax(ColorCM)), np.amax(OrientationCM)  ) )

        ### display results #################
        print 'IntensityCM_MAX',np.amax(IntensityCM)
        print 'ColorsCM_MAX',np.amax(ColorCM)
        print 'OrientationCM_MAX',np.amax(OrientationCM)
        print 'SM_MAX',np.amax(SM)
        print 'CM_MAX',maxCMval
        
        # cv2.imshow('SrcImg',img)
        # cv2.imshow('IntensityCM',IntensityCM/np.amax(IntensityCM))        
        # cv2.imshow('IntensityCM',IntensityCM/maxCMval)
        cv2.imshow('ColorCM',ColorCM/np.amax(ColorCM))
        # cv2.imshow('OrientationCM',OrientationCM/maxCMval)
        # cv2.imshow('SaliencyMap',SM/np.amax(SM))
        #####################################

    def GetIntensityCM(self,img):
        '''
           入力画像のIntensity Couspicuity Mapを求める
           argv: img         -> 入力画像
           dst : IntensityCM -> 強度の顕著性マップ
        '''
        # Get Intensity image # 出力はfloat
        IntensityImg = self.GetIntensityImg(img)
        # IntensityImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Create Gaussian pyramid　# [0]が元画像
        IntensityPyr  = self.GauPyr(IntensityImg) 

        # Get Feature Maps
        IntensityFM  = self.GetFM(IntensityPyr) 

        # Get Conspicuity Map
        print 'getintensitycm'
        IntensityCM = self.GetCM(IntensityFM) 

        cv2.imshow('IntensityImg',IntensityImg/np.amax(IntensityImg))
        # for i in range(0,4):
            # print 'max_intensityFM',np.amax(IntensityFM[i])
            # cv2.imshow('intensityFM[%s]'%i,IntensityFM[i]/np.amax(IntensityFM[0]))




        return IntensityCM

    def GetColorCM(self,img):        
        '''
           入力画像のColors Couspicuity Mapを求める
           argv: img     -> 入力画像
           dst : ColorCM -> Colorsの顕著性マップ
        '''
        # Get each Color images # 出力はfloat
        R,G,B,Y = self.GetRGBY(img) # float #ok

        # Create Gaussian Pyramids of each colors
        RPyr, GPyr, BPyr, YPyr = map(lambda x:self.GauPyr(x),[R,G,B,Y])
        
        # cv2.imshow('R',R/np.amax(R))
        # cv2.imshow('G',G/np.amax(G))
        # cv2.imshow('B',B/np.amax(B))
        # cv2.imshow('Y',Y.astype(np.uint8))
        # cv2.imshow('R',R/np.amax(R))
        # cv2.imshow('G',G/np.amax(G))
        # cv2.imshow('B',B/np.amax(B))
        # cv2.imshow('Y',Y/np.amax(Y))


        # Get Feature Maps of RG & BY
        RGFM = self.GetColorFM(RPyr,GPyr)
        BYFM = self.GetColorFM(BPyr,YPyr)
        print np.amax(RGFM[0])
        print np.amax(BYFM[0])

        for i in range(6):
            # cv2.imshow('RGFM%s'%i,RGFM[i]/np.amax(RGFM[i]))
            pass

        # cv2.imshow('R-G',RGFM[0]/np.amax(RGFM[0]))
        # cv2.imshow('B-Y',BYFM[0]/np.amax(RGFM[0]))
        # cv2.imshow('rg',RGFM[5]/np.amax(RGFM[5]))
        # cv2.imshow('R',RPyr[6]/np.amax(RPyr[0]))
        # cv2.imshow('G',GPyr[6]/np.amax(RPyr[0]))

        for i in range(6):
            # print RPyr[i].shape
            # cv2.imshow('RGFM%s'%i,RGFM[i]/np.amax(RGFM[0]))
            pass
        # Get Conspicuity Map
        RGCM = self.GetCM(RGFM)
        BYCM = self.GetCM(BYFM)
        ColorCM = RGCM + BYCM
        print np.amax(RGCM)
        print np.amax(BYCM)
        # cv2.imshow('RGCM',RGCM/np.amax(RGCM))
        # cv2.imshow('BYCM',BYCM/np.amax(RGCM))


        return ColorCM

    def GetOrientationCM(self,img):
        '''
           入力画像のOrientation Couspicuity Mapを求める
           argv: img     -> 入力画像
           dst : OrientationCM -> Colorsの顕著性マップ
        '''
        

        IntensityImg = self.GetIntensityImg(img)
        # cv2.imshow('IntensityImg',IntensityImg/np.amax(IntensityImg))

        # Get each Orientation images pyramid
        Orientation0Pyr, Orientation45Pyr, Orientation90Pyr, Orientation135Pyr = self.GetOrientationPyr(IntensityImg)
        
        # Create feature maps
        Orientation0FM = self.GetFM(Orientation0Pyr)
        Orientation45FM = self.GetFM(Orientation45Pyr)
        Orientation90FM = self.GetFM(Orientation90Pyr)
        Orientation135FM = self.GetFM(Orientation135Pyr)

        # create Couspicuity Map floatなのに注意！
        Orientation0CM = self.GetCM(Orientation0FM)
        Orientation45CM = self.GetCM(Orientation45FM)
        Orientation90CM = self.GetCM(Orientation90FM)
        Orientation135CM = self.GetCM(Orientation135FM)        
        OrientationCM     = Orientation0CM + Orientation45CM + Orientation90CM + Orientation135CM

        ### display results ###############################
        # for i in range(10):
            # OrientationImg = Orientation0Pyr[i] + Orientation45Pyr[i] + Orientation90Pyr[i] + Orientation135Pyr[i]
            # cv2.imshow('OrientationImg%s'%i,OrientationImg)
        ###################################################

        return OrientationCM

    def GetIntensityImg(self,img):
        '''
            入力画像のIntensityImgを返す
            argv: img ->  3chのColorImg
            dst : IntensityImg -> 1chのImg
        '''
        b,g,r = cv2.split(img)
        IntensityImg = b/3. + g/3. + r/3.
        
        ### display results #####################################
        # cv2.imshow('intensity',IntensityImg/np.amax(IntensityImg))
        #########################################################

        return IntensityImg

    def GetOrientationPyr(self,img):
        '''
            入力画像からorientationのfeature mapを生成して返す
            argv: img -> 入力画像，1ch,float
            dst : rg,by -> 正規化されたrgとbyの画像，!!!float!!!,1ch
        '''
        GauPyrImg = self.GauPyr(img) #10枚の画像
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

        return gabor0, gabor45, gabor90, gabor135

    def GetColorFM(self,pyr1,pyr2):
        '''
           入力された二つのGauPyrに対してFMを生成する
           階層が3つ離れているものと4つ離れているものの差分をとる
            args : pyr1,pyr2  ->   各階層のgaussian画像が入ったlist
            dst  : FM         ->   特定の階層のFMが入ったlist

        '''
        FM = range(6)        
        for i,s in enumerate(range(2,5)):
            # FM[2*i]   = cv2.absdiff( (pyr1[s] - pyr2[s]) , (pyr2[s+3] - pyr1[s+3]) )
            # FM[2*i+1] = cv2.absdiff( (pyr1[s] - pyr2[s]) , (pyr2[s+4] - pyr1[s+4]) )
            FM[2*i]   = cv2.absdiff( (pyr1[s] - pyr2[s]) , (pyr1[s+3] - pyr2[s+3]) )
            FM[2*i+1] = cv2.absdiff( (pyr1[s] - pyr2[s]) , (pyr1[s+4] - pyr2[s+4]) )
        return FM

    def GetRGBY(self,img):
        '''
            入力された画像を正規化して返す
        '''
        # rgbの正規化，他よりもその成分が大きいところを抽出
        b, g, r = cv2.split(img)
        # b,g,r = map(lambda x: x.astype(np.float),[b,g,r])
        
        B,G,R = map(lambda x,y,z: x*1. - (y*1. + z*1.)/2., [b,g,r],[r,r,g],[g,b,b])
        Y = (r*1. + g*1.)/2. - np.abs(r*1. - g*1.)/2. - b*1.
        # 負の部分は0にする
        R[R<0] = 0
        G[G<0] = 0
        B[B<0] = 0
        Y[Y<0] = 0

        # diaplay results ##########################################
        # cv2.imshow('r',r/np.amax(r))
        # cv2.imshow('g',g/np.amax(g))
        # cv2.imshow('b',b/np.amax(b))

        # cv2.imshow('R',R/np.amax(R))
        # cv2.imshow('G',G/np.amax(G))
        # cv2.imshow('B',B/np.amax(B))
        # cv2.imshow('Y',Y/np.amax(Y))
        ############################################################
        return R,G,B,Y

    def GauPyr(self,img):
        '''
            入力された画像のgaussian pyramidを計算する
            args : img    -> Mat,Uint8,1chの画像
            dst  : pyr  -> 元画像合わせて10階層のgaussian画像が入ったlist，それぞれargsと同じサイズ
                           pyr[i].shapeは，src.shape / 2**i
                   pyr2 -> 極端に差がでるpyr

        '''

        pyr  = range(10)
        pyr2 = range(10)

        pyr[0]  = img 
        pyr2[0] = img

        # Create Gaussian pyramid
        for i in range(1,10):
            pyr[i] = cv2.pyrDown(pyr[i-1])
            pyr2[i] = cv2.GaussianBlur(pyr[i-1], (9,9), 2**1)
        
        # Resize pyramid 
        for i in range(10):
            pyr[i] = cv2.resize(pyr[i],(512,512), interpolation = cv2.INTER_LINEAR)
            pyr2[i] = cv2.resize(pyr2[i],(512,512), interpolation = cv2.INTER_LINEAR)
            # print np.array(pyr[i]).shape
            # cv2.imshow('r%s'%i,pyr2[i]/np.amax(pyr[i]))
        return pyr2

    def GetFM(self,pyr):
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

    def GetCM(self,FM):
        '''
            入力されたFMから各特徴毎のConspicuity Map(顕著性map)を生成する
            args : FM  -> FMが入ったlist
            dst  : CM  -> 顕著性マップ，!!!float!!!
        '''
        normalizedFM = range(6)
        CM = np.zeros_like(FM[0])
        # 画像を正規化して，それぞれ足し合わせる
        for i in range(0,6):
            FM0to1 = self.RangeNormalize0to1(FM[i]) # 0-1となる正規化
            AveLocalMax = self.GetAveMaxima(FM0to1)
            normalizedFM[i] = FM0to1 * (1-AveLocalMax)**2 #FM x 0.3前後
            
            CM = CM + normalizedFM[i]
            print 'AveLocalMax%s'%i,AveLocalMax
            # cv2.imshow('FM%s'%i,FM[i]/np.amax(FM[0]))
            # cv2.imshow('normalizedFM%s'%i,normalizedFM[i])        
        return CM
        
    def RangeNormalize0to1(self,img):
        '''
            入力された画像を正規化して返す，最大値を1とする正規化
            args : img -> uint,1ch
            dst  : normalizedImg  -> 正規化された画像
        '''
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(np.array(img))

        if maxval != minval:
            normalizedImg = img/(maxval-minval) + minval/(minval-maxval)
        else:
            normalizedImg = img - minval


        return normalizedImg

    def GetAveMaxima(self,img):
        '''
            入力された画像の極大値の平均を求める
            args : img -> uint,1ch
            dst  : AveLocalMax　-> 極大値の平均
        '''
        '''maxima = np.array(signal.argrelmax(IntensityImg))'''

        maxima = np.array(signal.argrelmax(img))
        TotalMaxima = 1
        for i in range(len(maxima[1])):
            u = maxima[0,i]
            v = maxima[1,i]
            Maxima = img[u,v]
            TotalMaxima += Maxima
        if len(maxima[1]) == 0:
            AveMaxima = 0
        else:
            AveMaxima = TotalMaxima/len(maxima[1])

        return AveMaxima

    def ViewKernel2(self,kernel):
        '''
            入力されたkernelをヒートマップで表示する
        '''

        CMat = ConvolutionMatrix()

        # データの準備
        x = arange(0, CMat.cols, 1)
        y = arange(0, CMat.rows, 1)
        X, Y = meshgrid(x, y) 
        Z = CMat.GetXkernel(CMat.rows)
        
        # グラフの設定
        '''2D'''
        # plt.xlabel('pixel')
        # plt.ylabel('pixel')
        # plt.title('Kernel Size = %s,'%self.k+'  Sigma = %s,'%self.sigma+'  Theta = %s,'%0+'  Lambda = %s,'%self.lambd+'  Gamma = %s'%self.gamma)
        # plt.pcolor(X, Y, Z)
        # plt.colorbar()

        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # classよんでる？
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_title('Convolution Matrix')

        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',)
        # ax.plot_wireframe(X,Y,Z, cmap = cm.RdPu, rstride=2, cstride=2)

        # plt.imshow()
        # plt.draw(-1)
        plt.pause(-1)

    def ViewImage(self,img):
        '''
            入力画像を3Dで表示する
        '''
        # データの準備

        x = arange(0, len(img[0]), 1)
        y = arange(0, len(img[1]), 1)
        X, Y = meshgrid(x, y) 
        
        # plotしたいkernel
        Z = cv2.split(img)[0] / 100.
        print Z.shape,X.shape
        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # classよんでる？
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_zlim(0, 5)
        ax.set_title('Image')

        # ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',)
        ax.plot_wireframe(X,Y,Z, cmap = 'jet', rstride=5, cstride=5)


        # plt.pause(-1)

    def DetectStraightLine(self,img):
        '''
            直線のみ検出した画像を作る
        '''
        # 画像の準備
        img =  self.GetIntensityImg(img)
        cv2.imshow('img',img/np.amax(img))

        #　カーネルの準備
        CMat = ConvolutionMatrix()
        kernel = CMat.GetXkernel(CMat.rows) # ok
        
        # 畳み込み
        out = cv2.filter2D(img,cv2.CV_64F, kernel)
        out = np.array(out/np.amax(out))
        
        # 後処理
        # print out<0.5
        out[out<0.5] = 0
        # print np.amax(out)
        # 領域ごとで見ればもっと良くなりそう

        pyr = self.GauPyr(out)
        for i in range(10):
            # cv2.imshow('pyr%s'%i,pyr[i])
            pass
        return out



   
################################################################################
# main
################################################################################
if __name__ == '__main__':

    # class recognition
    SM = SaliencyMap()
    CMat = ConvolutionMatrix()


    ### get image #################
    src = sys.argv[1]
    img = cv2.imread(src)
    if img is None:
        print  '    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'\
              +'    !!!!!!There is not %s'%src + '!!!!!!!!\n'\
              +'    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        sys.exit()
    img = cv2.resize(img,(512,512))
    # img[np.sum(img,axis=2)==0] = 255
    ################################
    # kernelをグラフで表示
    # SM.ViewImage(img)
    # plt.imshow(img)
    # plt.pause(-1)
    # CMat.ViewKernel(SM.GaborKernel_0)
    # line = SM.DetectStraightLine(img)
    # cv2.imshow('line',line/np.amax(line))
    # Get Saliency Map
    SM.saliency_map(img)
    
    cv2.waitKey(-1)