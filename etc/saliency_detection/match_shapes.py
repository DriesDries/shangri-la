# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import log

class MatchShapes():

    def __init__(self):

        self.scale = range(10,1,-1)



    def MatchShapesMain(self,img,tmp):
        '''
            読み込んだ二つの画像の類似度をcv2.matchShapesを用いて求める
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
    
        # grayへの変換
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)        

        # tmp -> tmps
        tmps = self.GetTmps(tmp)


        # 閾値かけて二値化処理する，ここもいじれそう
        contour_img = self.ontour

        # 輪郭検出する

        # tmpsを移動させながらmatch，いいやついくつか検出

















        # この場合は閾値で2値化
        # ret , threshImg  = cv2.threshold(img , 240, 255,0) # retは平均もしくは中央値みたいな
        # for i in range(len(tmps)):
            # ret2, threshtmp = cv2.threshold(tmps[i], 127, 255,0)
        
        # 輪郭抜き出してる？
        contours,hierarchy = cv2.findContours(img,2,1)
        cnt1 = contours[0]
        contours,hierarchy = cv2.findContours(tmp,2,1)
        cnt2 = contours[0]


        # ret = cv2.matchShapes(cnt1,cnt2,1,0.0)

        # print 'result',retma

        # out = img
        # return out

    def GetTmps(self,tmp):
        '''
            入力されたtmpを複数のスケールにし，リストとして返す
            args :      -> tmp画像,3ch画像
            dst  :      -> self.scaleで指定されたスケールのtmp画像が入ったリスト
        '''

        tmps = range(len(self.scale))
        for i,s in enumerate(self.scale):
            tmps[i] = cv2.resize(tmp, (s, s))

        return tmps

    def MatchHuMoments(self,img,tmp):
        

        # extract moments
        moment_img = cv2.moments(img)
        moment_tmp = cv2.moments(tmp)

        # calculate Hu Moments
        hu_img = cv2.HuMoments(moment_img) # これでモーメントからHuモーメントを抽出
        hu_tmp = cv2.HuMoments(moment_tmp)

        # compare Hu Moments
        d = sum([abs(self.h2m(x1)-self.h2m(x2)) for (x1,x2) in zip(hu_img.T[0],hu_tmp.T[0])])
        # listにしなきゃだめ
        

        print d


    def h2m(self,x):
       if x==0:
          return 0.0
       elif x>0:
          return 1.0/log(x)
       else:
          return -1.0/log(-x)

    def MatchShapes(self,img,tmps):

        ret , thresh_img  = cv2.threshold(img, 240, 255,0) # retは平均もしくは中央値みたいな
        ret2, thresh_tmp = cv2.threshold(tmp, 127, 255,0)
        
        cv2.imshow('img2',img)
        cv2.imshow('tmp2',tmp)

        # 輪郭抜き出してる？
        contours,hierarchy = cv2.findContours(img,2,1) # imgも変わってしまう
        cnt1 = contours[0]
        contours,hierarchy = cv2.findContours(tmp,2,1)
        cnt2 = contours[0]

        ret = cv2.matchShapes(cnt1,cnt2,1,0.0)

        print ret


################################################################################
# メイン
################################################################################
if __name__ == '__main__':


    # read class
    MS = MatchShapes()

    # get image
    img = cv2.imread(sys.argv[1],0)
    # if img is None:
        # print '!'*20,'There is not %s'%sys.argv[1],'!'*20
        # sys.exit()
    tmp = cv2.imread(sys.argv[2],0)
    # if tmp is None:
        # print '!'*20,'There is not %s'%sys.argv[1],'!'*20
        # sys.exit()
    

    eimg = cv2.Canny(img, threshold1= 0, threshold2= 0,apertureSize = 3)
    etmp = cv2.Canny(tmp, threshold1= 30, threshold2= 200,apertureSize = 3)
    cv2.imshow('eimg',eimg)
    cv2.imshow('etmp',etmp)
    cv2.imshow('img',img)
    cv2.imshow('tmp',tmp)
    # matchshapes    
    # MS.MatchShapesMain(img,tmp)
    MS.MatchHuMoments(eimg,etmp)
    MS.MatchShapes(img,tmp)

    '''メモ
    
    MatchHuMomentsはgray scaleでの比較
    MatchShapesは輪郭での比較

    両方やる

    '''

    cv2.waitKey(-1)
    