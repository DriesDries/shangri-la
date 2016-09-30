# -*- coding: utf-8 -*-
'''
    A Model of Saliency-Based Visual Attention for Rapid Scene Analysis�Όgװ
    ���������SaliencyMap����ᣬ��������
    Usage: $ python saliency_map.py <argv> 
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

class SaliencyMap():

    def __init__(self): # print self.name��ʹ����
        
        '''Gabor Filer''' 
        self.k = 99 # odd
        self.ksize  = (self.k,self.k)
        self.sigma = 1
        self.lambd = 5
        self.gamma = 1
        self.GaborKernel_0   = cv2.getGaborKernel(ksize = self.ksize, sigma = self.sigma,theta = 0, lambd = self.lambd, gamma = self.gamma)
        self.GaborKernel_45  = cv2.getGaborKernel(self.ksize, self.sigma, 45, self.lambd,  self.gamma)
        self.GaborKernel_90  = cv2.getGaborKernel(self.ksize, self.sigma, 90, self.lambd,  self.gamma)
        self.GaborKernel_135  = cv2.getGaborKernel(self.ksize, self.sigma, 135, self.lambd,  self.gamma)       


    def saliency_map(self,img):
        '''
            �������줿�����Saliency Map������
            args : img  -> ��������
            dst  : SM   -> saliencymap,uint8��Mat��1ch����
            param: CM   -> ���؏՚�����Ҏ�����줿Couspicuity Map(�����map)
        '''
        # Get Conspicuity Map of each features
        IntensityCM = self.GetIntensityCM(img)
        ColorCM = self.GetColorCM(img)
        OrientationCM = self.GetOrientationCM(img)
        
        # create Saliency Map
        SM = IntensityCM/3. + ColorCM/2. + OrientationCM/6.



        ### display results #################
        print 'IntensityCM_MAX',np.amax(IntensityCM)
        print 'ColorsCM_MAX',np.amax(ColorCM)
        print 'OrientationCM_MAX',np.amax(OrientationCM)
        print 'SM_MAX',np.amax(SM)
        
        cv2.imshow('SrcImg',img)
        cv2.imshow('IntensityCM',((IntensityCM/np.amax(IntensityCM))*255).astype(np.uint8))
        cv2.imshow('ColorCM',( (ColorCM/np.amax(ColorCM))*255).astype(np.uint8))
        cv2.imshow('OrientationCM',OrientationCM/np.amax(OrientationCM))
        cv2.imshow('SaliencyMap',SM/np.amax(SM))
        #####################################

    def GetIntensityCM(self,img):
        '''
           ���������Intensity Couspicuity Map������
           argv: img         -> ��������
           dst : IntensityCM -> ���Ȥ�����ԥޥå�
        '''
        # Get Intensity image # ������float
        IntensityImg = self.GetIntensityImg(img)

        # Create Gaussian pyramid��# [0]��Ԫ����
        IntensityPyr  = self.GauPyr(IntensityImg) 

        # Get Feature Maps
        IntensityFM  = self.GetFM(IntensityPyr) 

        # Get Conspicuity Map
        IntensityCM     = self.GetCM(IntensityFM) 

        return IntensityCM

    def GetColorCM(self,img):        
        '''
           ���������Colors Couspicuity Map������
           argv: img     -> ��������
           dst : ColorCM -> Colors������ԥޥå�
        '''
        # Get each Color images # ������float
        R,G,B,Y = self.GetRGBY(img) # float #ok

        # Create Gaussian Pyramids of each colors
        RPyr, GPyr, BPyr, YPyr = map(lambda x:self.GauPyr(x),[R,G,B,Y])
        # Get Feature Maps of RG & BY
        RGFM = self.GetColorFM(RPyr,GPyr)
        BYFM = self.GetColorFM(BPyr,YPyr)
        
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

        return ColorCM

    def GetOrientationCM(self,img):
        '''
           ���������Orientation Couspicuity Map������
           argv: img     -> ��������
           dst : OrientationCM -> Colors������ԥޥå�
        '''
        

        IntensityImg = self.GetIntensityImg(img)
        # cv2.imshow('IntensityImg',IntensityImg/np.amax(IntensityImg))

        # Get each Orientation images pyramid
        Orientation0Pyr, Orientation45Pyr, Orientation90Pyr, Orientation135Pyr = self.GetOrientationPyr(IntensityImg)
        for i in range(10):
            OrientationImg = Orientation0Pyr[i] + Orientation45Pyr[i] + Orientation90Pyr[i] + Orientation135Pyr[i]
            # cv2.imshow('OrientationImg%s'%i,OrientationImg)
        # Create feature maps
        Orientation0FM = self.GetFM(Orientation0Pyr)
        Orientation45FM = self.GetFM(Orientation45Pyr)
        Orientation90FM = self.GetFM(Orientation90Pyr)
        Orientation135FM = self.GetFM(Orientation135Pyr)

        # create Couspicuity Map float�ʤΤ�ע�⣡
        Orientation0CM = self.GetCM(Orientation0FM)
        Orientation45CM = self.GetCM(Orientation45FM)
        Orientation90CM = self.GetCM(Orientation90FM)
        Orientation135CM = self.GetCM(Orientation135FM)        
        OrientationCM     = Orientation0CM + Orientation45CM + Orientation90CM + Orientation135CM

        return OrientationCM

    def GetIntensityImg(self,img):
        '''
            ���������IntensityImg�򷵤�
            argv: img ->  3ch��ColorImg
            dst : IntensityImg -> 1ch��Img
        '''
        b,g,r = cv2.split(img)
        IntensityImg = b/3. + g/3. + r/3.
        
        ### display results #####################################
        # cv2.imshow('intensity',IntensityImg/np.amax(IntensityImg))
        #########################################################

        return IntensityImg

    def GetOrientationPyr(self,img):
        '''
            �������񤫤�orientation��feature map�����ɤ��Ʒ���
            argv: img -> ��������1ch,float
            dst : rg,by -> ��Ҏ�����줿rg��by�λ���!!!float!!!,1ch
        '''
        GauPyrImg = self.GauPyr(img) #10ö�λ���
        gabor0  = range(10)
        gabor45 = range(10)
        gabor90 = range(10)
        gabor135= range(10)        
        
        # ���줾��gray��gaupyr��gabor filter�ˤ�������� # uint8
        for i in range(10):
            gabor0[i]   = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_0)
            gabor45[i]  = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_45)
            gabor90[i]  = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_90)
            gabor135[i] = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_135)
            # gabor180[i] = cv2.filter2D(GauPyrImg[i], cv2.CV_8U, self.GaborKernel_180)

        return gabor0, gabor45, gabor90, gabor135

    def GetColorFM(self,pyr1,pyr2):
        '''
           �������줿���Ĥ�GauPyr�ˌ�����FM�����ɤ���
           �A�Ӥ�3���x��Ƥ����Τ�4���x��Ƥ����Τβ�֤�Ȥ�
            args : pyr1,pyr2  ->   ���A�Ӥ�gaussian������ä�list
            dst  : FM         ->   �ض����A�Ӥ�FM����ä�list

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
            �������줿�������Ҏ�����Ʒ���
        '''
        # rgb����Ҏ���������⤽�γɷ֤��󤭤��Ȥ������
        b, g, r = cv2.split(img)
        # b,g,r = map(lambda x: x.astype(np.float),[b,g,r])
        
        B,G,R = map(lambda x,y,z: x*1. - (y*1. + z*1.)/2., [b,g,r],[r,r,g],[g,b,b])
        Y = (r*1. + g*1.)/2. - np.abs(r*1. - g*1.)/2. - b*1.
        # ؓ�β��֤�0�ˤ���
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
            �������줿�����gaussian pyramid��Ӌ�㤹��
            args : img    -> Mat,Uint8,1ch�λ���
            dst  : pyr  -> Ԫ����Ϥ碌��10�A�Ӥ�gaussian������ä�list�����줾��args��ͬ��������
                           pyr[i].shape�ϣ�src.shape / 2**i
                   pyr2 -> �O�ˤ˲�Ǥ�pyr

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
           �������줿GauPyr�ˌ�����DoG��Ӌ�㤹��
           �A�Ӥ�3���x��Ƥ����Τ�4���x��Ƥ����Τβ�֤�Ȥ�
            args : pyr     ->   ���A�Ӥ�gaussian������ä�list
            dst  : DoG     ->   �ض����A�Ӥ�DoG����ä�list

        '''
        # s��scale
        FM = range(6)
        for i,s in enumerate(range(2,5)):
            FM[2*i]  = cv2.absdiff(pyr[s],pyr[s+3])
            FM[2*i+1] = cv2.absdiff(pyr[s],pyr[s+4])

        return FM

    def GetCM(self,FM):
        '''
            �������줿FM������؏՚���Conspicuity Map(�����map)�����ɤ���
            args : FM  -> FM����ä�list
            dst  : CM  -> ����ԥޥåף�!!!float!!!
        '''
        normalizedFM = range(6)
        CM = np.zeros_like(FM[0])
        # �������Ҏ�����ƣ����줾���㤷�Ϥ碌��
        for i in range(0,6):
            FM0to1 = self.RangeNormalize0to1(FM[i]) # 0-1�Ȥʤ���Ҏ��
            AveLocalMax = self.GetAveMaxima(FM0to1) # �����ĤʘO�󂎤�ƽ���� 0.4ǰ��
            normalizedFM[i] = FM0to1 * (1-AveLocalMax)**2 #FM x 0.3ǰ��
            
            CM = CM + normalizedFM[i]
            # print AveLocalMax
            # print np.amax(FM0to1)
            # cv2.imshow('FM%s'%i,FM[i]/np.amax(FM[i]))
            # cv2.imshow('normalizedFM%s'%i,normalizedFM[i]/np.amax(FM0to1))        
        return CM
        
    def RangeNormalize0to1(self,img):
        '''
            �������줿�������Ҏ�����Ʒ�������󂎤�1�Ȥ�����Ҏ��
            args : img -> uint,1ch
            dst  : normalizedImg  -> ��Ҏ�����줿����
        '''
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(np.array(img))

        if maxval != minval:
            normalizedImg = img/(maxval-minval) + minval/(minval-maxval)
        else:
            normalizedImg = src - minval

        return normalizedImg

    def GetAveMaxima(self,img):
        '''
            �������줿����ΘO�󂎤�ƽ��������
            args : img -> uint,1ch
            dst  : AveLocalMax��-> �O�󂎤�ƽ��
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

    def ViewKernel(self,kernel):
        '''
            �������줿kernel��ҩ`�ȥޥåפǱ�ʾ����
        '''
        # �ǩ`���Μʂ�
        x = arange(0, self.k, 1)
        y = arange(0, self.k, 1)
        X, Y = meshgrid(x, y) 
        
        # plot������kernel
        Z = np.array(self.GaborKernel_0)
    

        # ����դ��O��
        '''2D'''
        # plt.xlabel('pixel')
        # plt.ylabel('pixel')
        # plt.title('Kernel Size = %s,'%self.k+'  Sigma = %s,'%self.sigma+'  Theta = %s,'%0+'  Lambda = %s,'%self.lambd+'  Gamma = %s'%self.gamma)
        # plt.pcolor(X, Y, Z)
        # plt.colorbar()

        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # class���Ǥ룿
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_title('Gabor Filter Kernel\n Kernel Size = %s,'%self.k+'  Sigma = %s,'%self.sigma+'  Theta = %s,'%0+'  Lambda = %s,'%self.lambd+'  Gamma = %s'%self.gamma)

        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',)
        # ax.plot_wireframe(X,Y,Z, cmap = cm.RdPu, rstride=2, cstride=2)


        plt.pause(.0001)

    def ViewImage(self,img):
        '''
            ���������3D�Ǳ�ʾ����
        '''
        # �ǩ`���Μʂ�

        x = arange(0, len(img[0]), 1)
        y = arange(0, len(img[1]), 1)
        X, Y = meshgrid(x, y) 
        
        # plot������kernel
        Z = img

        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # class���Ǥ룿
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_title('Image')

        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',)
        # ax.plot_wireframe(X,Y,Z, cmap = cm.RdPu, rstride=2, cstride=2)


        plt.pause(.0001)
   
################################################################################
# main
################################################################################
if __name__ == '__main__':

    sm = SaliencyMap()
    
    ### get image #################
    src = sys.argv[1]
    img = cv2.imread(src)
    if img is None:
        print  '===================================================\n'\
              +'=========There is not %s'%src + '==============\n'\
              +'==================================================='
        sys.exit()
    img = cv2.resize(img,(512,512))
    # img =255 - img
    ################################
    # kernel�򥰥�դǱ�ʾ

    # sm.ViewKernel(sm.GaborKernel_0)
    
    # Get Saliency Map
    sm.saliency_map(img)
    
    cv2.waitKey(-1)