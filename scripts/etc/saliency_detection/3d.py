# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
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

class name():

    
    def __init__(self):
        
        self.value = 0


    def ViewImage(self,img):
        '''
            入力画像を3Dで表示する
        '''
        # データの準備

        x = arange(0, len(img[0]), 1)
        y = arange(0, len(img[1]), 1)
        X, Y = meshgrid(x, y) 
        
        # plotしたいkernel
        Z = cv2.split(img)[0]
        print Z.shape,X.shape
        '''Axes3D'''
        fig = plt.figure()
        ax = Axes3D(fig)    # classよんでる？
        # ax.set_xlabel('pixel')
        # ax.set_ylabel('pixel')        
        # ax.set_zlabel('intensity')
        # ax.set_zlim(0, 1)
        # ax.set_title('Image')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap = 'jet',linewidth=0)
        # ax.plot_wireframe(X,Y,Z, cmap = 'Greys', rstride=5, cstride=5)

        plt.pause(-1)


    def module2(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''

        out = img
        return out

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


################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    cls = name()

    # img = cv2.imread('../../../image/rinko/r.png')
    # print np.amax(img),img.shape
    # img = cv2.imread('../../../image/rinko/dust.jpg')

    # 画像の用意
    # img = np.ones((512,512,3))
    # # cv2.circle(img,(255,255),50,(255,255,255),thickness=-1)
    # # img = img.astype(np.uint8)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(512,512))

    # img[255,255] = 255
    # # imgs = range(10)
    # # imgs[0] = img



    # img2 = cv2.pyrDown(img)
    # img3 = cv2.pyrDown(img2)
    # img4 = cv2.pyrDown(img3)
    # img5 = cv2.pyrDown(img4)
    # img6 = cv2.pyrDown(img5)
    # img7 = cv2.pyrDown(img6)
    # img8 = cv2.pyrDown(img7)
    # img9 = cv2.pyrDown(img8)


    # # img = cv2.resize(img7,(512,512))

    # # cv2.imshow('img9',img6)



    # cm1 = cv2.imread('./cmma.png')
    # cm2 = cv2.imread('./cmmr.png')
    # cm3 = cv2.imread('./cmmb.png')
    # cm4 = cv2.imread('./cmmg.png')    



    # # # cv2.imshow('r',cm2)
    # cm = 0.8*cm1.astype(np.float32)+cm2.astype(np.float32)+1.2*cm3.astype(np.float32)+cm4.astype(np.float32)
    # cm = cv2.cvtColor(cm,cv2.COLOR_BGR2GRAY)
    # print cm.shape
    # for i in range(30):
    #     cm = np.delete(cm,0,1)
    # print cm.shape
    # cm = cv2.resize(cm,(512,512))
    # cv2.imshow('cm',cm/np.amax(cm))



    img = cv2.imread('../../../image/rinko/dust.jpg')
    img = cv2.resize(img,(512,512))
    cv2.imshow('img',img)
    b,g,r = cv2.split(img)
    cv2.imshow('r',r)
    cv2.imshow('g',g)
    cv2.imshow('b',b)
    z = np.zeros_like(r)

    img = cv2.merge((z,z,r))
    b = cv2.merge((b,z,z))
    g = cv2.merge((z,g,z))
    cv2.imshow('b',b)
    cv2.imshow('g',g)

    # R = r*1.- (g*1. + b*1.)/2.
    # R = b*1.- (r*1. + g*1.)/2. + 50

    # B,G,R = map(lambda x,y,z: x*1. - (y*1. + z*1.)/2., [b,g,r],[r,r,g],[g,b,b])
    # Y = (r*1. + g*1.)/2. - np.abs(r*1. - g*1.)/2. - b*1.
    # 負の部分は0にする
    # R[R<0] = 0
    # G[G<0] = 0
    # B[B<0] = 0
    # Y[Y<0] = 0
    # print np.amax(R)

    # cv2.imshow('R',R/np.amax(R))

    # cv2.imshow('img',img)
    # for i in range(10):
        # print i
        # img = cv2.pyrDown(imgs[i])
        # img = cv2.resize(img,(512,512),interpolation = cv2.INTER_LINEAR)    
        # if i ==9:
            # break
        # imgs[i+1] = img
        # cv2.imshow('img%s'%i,img)
    # edge = cv2.imread('./size2.png')
    # edge = cv2.Canny(edge, threshold1= 180, threshold2= 250,apertureSize = 3)
    # cv2.imshow('edge',edge)



    # print img.shape
    # cv2.imshow('img',img)


    # ガウシアンフィルタ
    # img = cv2.GaussianBlur(img,(5,5),2**8)
    # cv2.imshow('gau',img)

    # if img is None:
        # print  '!!!!!!There is not %s'%sys.argv[1] + '!!!!!!!!'
        # sys.exit()
    

    # cls.ViewImage(cm/np.amax(cm))
    

    # cv2.imshow('Input',  img)
    # cv2.imshow('Output', out)
    cv2.waitKey(-1)
    