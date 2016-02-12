    # -*- coding: utf-8 
'''
    Template Matchingの実装
    Usage: $ python template.py <argv[1]> <argv[2]>
    argv[1]: 検索される画像のpath 
    argv[2]: 検索する画像のpath
    dst    : matches
             res_img
    param  : 
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pylab import *
import cmath
import math
from scipy import signal
import glob

class TemplateMatching():

    def __init__(self):

        # imgの大きさ
        self.size = (512,512)

        # tmpの大きさ，ピクセル数
        self.scale = np.arange(300,50,-2)

        # いくつまでの丸を検出するか
        self.detectNumber = 2

        # sharpen filter kernel
        self.sharpenKernel = np.array([[0 ,-1, 0],
                                       [-1, 5,-1],
                                       [0 ,-1, 0]])
        # template matching method
        self.method = cv2.TM_CCORR
        # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,cv2.TM_CCORR_NORMEDのどれか
        # それぞれの説明はスライド参照

    def TMmain(self,img,tmp):
        '''
            TMのメイン処理を行う
            args : img      -> 検索される画像, 3ch
                   tmp      -> template画像, 3ch
            dst  : matches  ->
            param: res_img  -> 最もマッチした部分を描画した画像
                   tmps     -> self.scaleで指定した大きさの複数の画像が入ってるリスト
                   edges_*  -> 入力画像もしくは画像のリストからedgeを抽出した画像もしくはリスト
        '''

        '''前処理'''
        # resize img & tmp
        resize_img = cv2.resize(img, self.size)
        resize_tmp = cv2.resize(tmp, self.size)

        # tmpをself.scaleで指定されたスケールの配列に
        tmps = self.GetTmps(resize_tmp)

        # get edge image
        edges_img  = self.GetEdges(resize_img) 
        edges_tmps = self.GetEdges(tmps) 

        # edge_img = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
        # edge_tmps = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)



        '''TM'''
        match,mask,mask2 = self.TemplateMatching(edges_img,edges_tmps)
        # mask2 = cv2.add(resize_img,mask)
        # cv2.imshow('mask2',mask2)


        # show results###########################
        # print np.array(tmps[0]).shape,np.array(tmps[len(self.scale)-1]).shape
        cv2.imshow('resize_img',resize_img)
        # cv2.imshow('edges_',edges_img)
        # cv2.imshow('match',match/np.amax(match))
        # cv2.imshow('mask',mask)
        cv2.imshow('edge_template',edges_tmps[0])
        # cv2.imshow('res_img',res_img)
        # print self.scale
        # print 'len',len(self.scale)
        for i in range(len(self.scale)):
            # cv2.imshow('%s'%i,matches[i]/np.amax(matches[i]))
            # cv2.imshow('%s'%i,edges_tmps[i])
            # print len(edges_tmps[i])
            pass
        ############################################

    def GetTmps(self,tmp):
        '''
            入力されたtmpを複数のスケールにし，リストとして返す
            args :      -> tmp画像,3ch画像
            dst  :      -> self.scaleで指定されたスケールのtmp画像が入ったリスト
            param:      -> 
        '''
        tmps = range(len(self.scale))
        for i,s in enumerate(self.scale):
            tmps[i] = cv2.resize(tmp, (s, s))

        return tmps

    def GetEdges(self,imgs):
        '''
            入力されたimgのedgeを入手して返す
            args : imgs   -> 画像, もしくは画像のリスト
            dst  : edges  -> 入力画像のedgeが入った画像，もしくはリスト
            param: threshold1 -> 低い方の閾値
                   threshold2 -> わからん
                   apertureSize -> 繋がっていないエッジの補完に関するparam
        '''
        # imgかtmpか
        if len(imgs) == len(self.scale):
            
            '''tmpに対する処理'''
            for i in range(len(self.scale)):
                # imgs[i] = cv2.GaussianBlur(imgs[i], (9,9), 2**1)
                imgs[i] = cv2.medianBlur(imgs[i],9)
                imgs[i] = cv2.filter2D(imgs[i], cv2.CV_8U, self.sharpenKernel)
                imgs[i] = cv2.Canny(imgs[i], threshold1= 90, threshold2= 200,apertureSize = 3)

        else:

            '''imgに対する処理'''
            # imgs = cv2.GaussianBlur(imgs, (9,9), 2**1)
            imgs = cv2.medianBlur(imgs,9)
            imgs = cv2.filter2D(imgs, cv2.CV_8U, self.sharpenKernel)
            imgs = cv2.Canny(imgs, threshold1= 90, threshold2= 200,apertureSize = 3)

        edges = imgs
        return edges

    def TemplateMatching(self,img,tmps):
        '''
            入力された画像でtemplate　matchingをする
            args : img      ->　探索される画像
                   tmp      -> 探索する画像の複数スケールが入ったリスト
            dst  : matches  -> TMの結果が入った画像のリスト,!!!サイズがそれぞれ異なる!!!
                   mask     -> TMの結果を描画した画像,mask.shape=(s,s,3),s=self.size        
                   max_vals -> max_vals[i]=np.amax(matches[i])
                   max_locs -> max_valsの座標
                   maxima   -> 描画した複数の資格の最大値一覧
            func : GetMatch -> TMをする          
        '''

        # Template Matching        
        matches, max_vals,max_locs = self.GetMatch(img,tmps)
        

        # normalized # tmps[i]に含まれるedgeのピクセル数で正規化(edgeの円周みたいな)
        for i in range(len(matches)):
            matches[i] = matches[i] / len(tmps[i][tmps[i]==255])
        
        # 一番だけ表示
        # mask, max_i = self.DrawMatchSquare(matches,max_vals,tmps)

        # self.detectNumberだけ表示
        mask, maxima = self.GetTMCM(matches,max_vals,tmps)

        # show results #######
        img = cv2.add(img,mask)
        cv2.imshow('result',img)
        # print max_i

        return 0,0,0

    def GetTMCM(self,matches,matches_max_vals,tmps):
        '''
            maskを作成して，そのマスクのところは計算しない
            u,vで取得されるとことv,uで指定しないとだめなとこある
        '''

        # 初期化        
        mask  = np.zeros(self.size) + 255            # 判別用
        mask2 = np.zeros(self.size).astype(np.uint8) # return用        
        val = range(len(matches))                    # matchesの各画像の最大値一覧
        uv = range(len(matches))                     # それらの最大値の座標一覧
        maxima = range(self.detectNumber)            # 描画された最大値の値
        draw = 0                                     # 描画された四角の数
        overlap = 0

        print 'keyが入力されたら終了します'

        for a in range(100000):
            '''matchesの最大値とその座標の抽出''''''resizeして3次元配列にすれば処理楽'''
            # matches一枚一枚からの最大値とその座標の抽出，格納
            for i in range(len(matches)):
                _,val[i],_,uv[i] = cv2.minMaxLoc(np.array(matches[i]))

            # matches全体からの最大値とその座標の抽出
            _,max_val,_,n = cv2.minMaxLoc(np.array(val))
            max_uv = tuple(uv[n[1]])

            '''max_uvの座標とそれに対応するtmpの角と中心の座標の取得'''
            top_left     = max_uv
            bottom_right = (top_left[0] + len(tmps[n[1]]) -1, top_left[1] + len(tmps[n[1]] -1))
            bottom_left  = (bottom_right[0], top_left[1])
            top_right    = (top_left[0], bottom_right[1])
            center       = ((top_left[1]+bottom_right[1])/2, (top_left[0]+bottom_right[0])/2 )

            '''重複と，tmpが画像からはみ出していないかの判別'''
            ans = bottom_right[0]<=511 and bottom_right[1]<=511 and mask[top_left] != 0 and mask[bottom_right] != 0 and mask[top_right] != 0 and mask[bottom_left] != 0 and mask[center] != 0


            if ans == True :
                print a,'描画'
                '''重複しないから描画する'''
                cv2.rectangle(mask, top_left, bottom_right, 0, thickness = -1)
                cv2.rectangle(mask2, top_left, bottom_right, 255, thickness = 2)
                cv2.putText(mask2,'rank%s'%(draw+1),(top_left[0]+5,top_left[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,255)
                
                maxima[draw] = max_val
                draw += 1

                # displat results ##############################################
                # print 'new',top_left,top_right,bottom_left,bottom_right, center
                # print mask[top_left], mask[bottom_right], mask[top_right], mask[bottom_left],mask[center]
                # print max_val
                # print overlap
                overlap = 0
                ################################################################

            else:   # 重複しているから描画しない
                overlap += 1
                pass
            # 描画した最大値を0にする
            matches[n[1]][max_uv[1],max_uv[0]] = 0

            if 'q' == cv2.waitKey(1):
                break

            # 指定した数だけdrawしたらbreakする
            if draw == self.detectNumber:
                break

        # show results ###########
        print 'draw',draw,'/',(a+1)
        print maxima
        #############################
        
        return mask2, maxima     

    def GetMatch(self,img,tmps):

        # 準備
        matches = range(len(self.scale))
        max_vals = range(len(self.scale))
        max_locs = range(len(self.scale))
        # template matching
        for i in range(len(self.scale)):

            matches[i] = cv2.matchTemplate(img,tmps[i],self.method)
            _,max_vals[i],_,max_locs[i] = cv2.minMaxLoc(np.array(matches[i]))

        return matches, max_vals, max_locs

    def DrawMatchSquare(self,matches,matches_max_vals,tmps):
        '''
            一番マッチしてる四角を表示したいとき
            matches_max_vals       matchesの各画像中で最大の値が入った配列
            max_pixel_val          matchesの全画像中で最大の値
            max_i                  matches内でmax_pixelがある要素の番号
            max_pixel_coordinate   画像中でのmax_pixel_valの座標
        '''

        # 準備        
        # mask = cv2.cvtColor(np.zeros(self.size).astype(uint8), cv2.COLOR_GRAY2BGR)
        mask = np.zeros(self.size).astype(np.uint8)

        # 一番マッチしてるtmpの番号
        _, _, _, max_i = cv2.minMaxLoc(np.array(matches_max_vals))
        max_i = max_i[1]
        _, max_pixel_val, _, max_pixel_coordinate = cv2.minMaxLoc(matches[max_i])

        # draw result
        top_left = max_pixel_coordinate
        bottom_right = (top_left[0] + len(tmps[max_i]), top_left[1] + len(tmps[max_i]))
        cv2.rectangle(mask, top_left, bottom_right, 255, thickness = 2, lineType = 8)

        return mask, max_i

################################################################################
# メイン
################################################################################
if __name__ == '__main__':

    # read class
    tm = TemplateMatching()

    # get image ##########################
    img = cv2.imread(sys.argv[1])
    if img is None:
        print '!'*20,'There is not %s'%sys.argv[1],'!'*20
        sys.exit()
    tmp = cv2.imread(sys.argv[2])
    if tmp is None:
        print '!'*20,'There is not %s'%sys.argv[1],'!'*20
        sys.exit()
    ######################################

    tmp = tmp[0:len(tmp[0])/2, 0:len(tmp[1])/2]

    # template matching
    tm.TMmain(img,tmp)

    cv2.waitKey(-1)
    