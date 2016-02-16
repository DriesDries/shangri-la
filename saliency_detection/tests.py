    # -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pylab import *
import cmath
import math
# from scipy import signal
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata


class TestClass():

    def __init__(self):
        self.sharpenKernel = np.array([[0 ,-1, 0],
                                       [-1, 5,-1],
                                       [0 ,-1, 0]])

    def HarrisCorner(self,img):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
        
        # img = cv2.resize(img,(648,1170))
        src = img
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('srcgray',gimg)
        
        # 前処理 # uint8
        # gimg = cv2.GaussianBlur(gimg, (9,9), 2**1)
        gimg = cv2.medianBlur(gimg,5)
        gimg = cv2.filter2D(gimg, cv2.CV_8U, cls.sharpenKernel)
        print np.array(gimg).dtype,np.array(gimg).shape
        cv2.imshow('filter',gimg)
        # edges = cv2.Canny(gimg,150,200,apertureSize = 3)   
        


        # harris
        dst = cv2.cornerHarris(gimg, blockSize = 2, ksize = 3, k = 0.04)
        dst = cv2.dilate(dst,None)

        img[dst>0.01*dst.max()]=[0,0,255]
        # display results ##################
        # print np.amax(dst)
        cv2.imshow('img',img)    
        cv2.imshow('dst',dst/np.amax(dst))
        ####################################

    def ImageIntegration(self,path):
        '''
            処理の概要
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
        os.chdir(path)
        print  os.getcwd()
        imgs = np.zeros((2160,1,3))
        deleMax = range(840)
        deleMax2 = range(2160,3000,1)
        for i, filename in enumerate(os.listdir('.')):
            # 画像読み込んでresize
            img = cv2.imread(filename)
            img = np.delete(img,deleMax,1)
            img = np.delete(img,deleMax2,1)

            if 0 <= i <= 2:
                if i == 0:
                    imgs = img
                else:
                    imgs = np.hstack((imgs,img))
            if 3 <= i <= 5:
                if i == 3:
                    imgs2 = img
                else:
                    imgs2 = np.hstack((imgs2,img))                    
            if 6 <= i <= 8:
                if i == 6:
                    imgs3 = img
                else:
                    imgs3 = np.hstack((imgs3,img))            
            print filename
            
        IMG = np.vstack((imgs,imgs2))
        IMG = np.vstack((IMG,imgs3))
        IMG = cv2.resize(IMG,(800,800))
        cv2.imshow('IMG',IMG)
        print IMG.shape

            # if i >= 3:

    def TemplateMatching(self, img, tmp):
        '''
            入力された画像とテンプレート画像でtemplate matchingを行う
        '''
        # edgeでやるとき
        # gimg = cv2.Canny(img, threshold1= 100, threshold2= 200,apertureSize = 3)         
        # tmp = cv2.Canny(tmp, threshold1= 100, threshold2= 200,apertureSize = 3) 
        
        # 普通にやるとき
        gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)

        gimg2 = gimg
        
        

        rows = len(tmp[:,0]) 
        cols = len(tmp[0]) 
        
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED']
        # method毎で行う
        for i, meth in enumerate(methods):
            gimg = gimg2
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(gimg,tmp,method)

            # 最小値，最大値，その座標
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                # top_left = min_loc
            # else:
                # top_left = max_loc
            
            # draw color
            # if i == 3:
                # color = [0,0,0]
            # else:
                # color = [0,0,0]
                # color[i] = 255
            color = 255
            # rectangle result
            top_left = max_loc
            bottom_right = (top_left[0] + rows, top_left[1] + rows)
            


            cv2.rectangle(img,(top_left[0],top_left[1]), bottom_right, color, 2)
            cv2.putText(img,meth,(top_left[0],top_left[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,color)
            cv2.imshow(meth,res/np.amax(res))
            cv2.imshow('Srcimg',img)
            cv2.imshow('template',tmp)
            cv2.imshow('GrayImg',gimg)

            print max_val,min_val
            # res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
            
            # b,g,r = cv2.split(res)
            # color = [b,g,r]
            # if i == 3:
            #     pass
            # else:
            #     color[i] = np.zeros_like(b) 
            # res = cv2.merge(color)

                
    def ResizeImg(self,path):

        os.chdir(path)
        os.getcwd()
        dele = range(840)
        dele2 = range(2160,3000,1)
        imgs = range(9)
        for i,filename in enumerate(glob.glob('*.JPG')):
            print filename,i
            img = cv2.imread(filename)
            img = np.delete(img,dele,1)
            img = np.delete(img,dele2,1)
            img = cv2.resize(img,(512,512))
            # imgs[i] = img
            # cv2.imshow('resized_rock_img%s.png'%i,img)
            cv2.imwrite('resized_rock_img%s.png'%i,img)

    def filter_1d(self):
        '''
        '''

        f = range(100)
        x2 = range(100)
        x = np.arange(0,np.pi,np.pi/10)
        sin = np.sin(x)


        f[0:10] = sin
        f[10:20] = sin
        f[20:30] = sin
        f[20:80] = np.zeros_like(f[20:80])   
        f[80:90] = sin
        f[90:100] = np.zeros_like(f[90:100])        

        plt.plot(x2,f)
        plt.pause(-1)
        # f[0:10] = 

    # def tests(self,img):

################################################################################
# メイン
################################################################################
if __name__ == '__main__':


    cls = TestClass()

    cls.filter_1d()


    # read img
    # img = cv2.imread(sys.argv[1])
    # cv2.imshow('img',img)
    # img2 = cv2.imread(sys.argv[2],0)

    # ret, thresh = cv2.threshold(img1, 127, 255,0)
    # ret, thresh2 = cv2.threshold(img2, 127, 255,0)

    # contours,hierarchy = cv2.findContours(thresh, mode = cv2.RETR_CCOMP, method = 1)
    # cnt1 = contours[0]
    # contours,hierarchy = cv2.findContours(thresh2,2,1)
    # cnt2 = contours[0]

    # ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
    # print ret
    # print np.array(cnt2).shape
    # # cv2.imshow('cnt1',cnt1)

    # m = cv2.moments(img) # これでモーメントを抽出
    # hu = cv2.HuMoments(m) # これでモーメントからHuモーメントを抽出

    # print np.array(m)
    # print np.array(m).shape
    # print np.array(m).dtype
    # print



    # a = np.array([1,2,3])
    # a = np.resize(a,(1,3))
    # print a
    # print a.T #転置行列，
    # display results ###############
    # print img.shape,template.shape
    # cv2.imshow('SrcImg',img)
    # cv2.imshow('template',template)
    #################################
    
    # path = './image/rock'
    # moduleの選択
    # cls.HarrisCorner(img)
    # cls.ImageIntegration(path)
    # cls.ResizeImg(path)
    
    # npts = 200
    # x = uniform(-2, 2, npts)
    # y = uniform(-2, 2, npts)
    # z = x*np.exp(-x**2 - y**2)

    # print x




    # # define grid.
    # xi = np.linspace(-2.1, 2.1, 100)
    # yi = np.linspace(-2.1, 2.1, 200)
    # # grid the data.
    # zi = griddata(x, y, z, xi, yi, interp='linear')
    # # contour the gridded data, plotting dots at the nonuniform data points.
    # CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    # CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
    #                   vmax=abs(zi).max(), vmin=-abs(zi).max())
    # plt.colorbar()  # draw colorbar
    # # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5, zorder=10)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # plt.title('griddata test (%d points)' % npts)
    # plt.show()




    cv2.waitKey(-1)
    