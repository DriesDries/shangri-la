# -*- coding: utf-8 -*-
'''
Region Growing Algolithmに基づいて画像を分割する
Usage: $ python region_growing.py 
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

class RegionGrowing():
    
    def __init__(self):
        
        # パラメータ
        self.img_size = 200  # 処理するときの画像サイズ
        self.thresh = 15     # 結合条件の閾値
        
        self.sample = np.array([[240,241,242,  0],
                                [254,245,246,  0],
                                [248,  0,250,  0],
                                [  0,  0,  0,  0]])

    def mouse_event(self,event, x, y, flags, param):
        '''
            画像をクリックすると引数がこの関数に引き渡されて、
            さらにRegionGrowingへと代入される
            args : event -> クリック
                   x,y   -> クリックされた座標
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(img, (x, y), 50, (0, 0, 255), -1)
            print 'seed x = '+str(x)+', y = '+str(y)
            self.RegionGrowing(img, x, y)

    def RegionGrowing(self,src,x,y):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''
        # 画像の準備,float32にする
        img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(self.img_size,self.img_size)).astype(np.float32)

        # seedの設定
        x = x*self.img_size/512
        y = y*self.img_size/512
        # seed = np.zeros([10000,2]).astype(np.uint8)
        seed = []
        seed.append([x,y])
        count = 1
        
        # region mapの準備
        region_map = np.zeros([self.img_size,self.img_size]).astype(np.uint8)
        region_map[y,x] = 255

        # start時の時間計測
        start = time.time()
        
        # region growing
        for i in xrange(100000):

            # 分割が終わったら終了
            if i == len(seed):
                break

            # renew seed point
            x,y = seed[i]
            
            # 画像の更新
            if i%40 == 0:
                cv2.imshow('growing',cv2.resize(region_map,(300,300)))
                cv2.waitKey(2)

            # 1pixelごとの処理
            for u in [x-1,x,x+1]:
                for v in [y-1,y,y+1]:
                    if -self.thresh < img[v,u] - img[y,x] and img[v,u] - img[y,x] < self.thresh:  # 領域結合の条件式

                        # 処理中断条件
                        if u == x and v == y: # seed pointだったらbreak
                            break
                        if u == 0 or u == self.img_size-1: # 画像の端だったらbreak
                            break
                        if v == 0 or v == self.img_size-1:
                            break
                        
                        # renew region map
                        region_map[v,u] = 255

                        # 重複してなければseed pointを保存する
                        for old_uv in seed: # 重複
                            if u == old_uv[0] and v == old_uv[1]:
                                break
                        else:           # 重複してない,forを抜けた後に実行される
                            seed.append([u,v])
                            count += 1
        # 終了時の時間入手
        processing_time  = time.time() - start
        print ('processing_time:{0}'.format(processing_time)) + '[sec]'

        # 結果の表示
        cv2.imshow('region_map',cv2.resize(region_map,(512,512)))
        masked_img = cv2.bitwise_and(src,src,mask =cv2.resize(region_map,(512,512)))
        cv2.imshow('masked_img',masked_img.astype(np.uint8))
        return 0



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
# main
################################################################################
if __name__ == '__main__':

    # class recognition
    rg = RegionGrowing()
    
    # get image
    # img = cv2.resize(cv2.imread('../../image/sample/water_coins.jpg'),(512,512))
    img = cv2.resize(cv2.imread('../../image/rock/sol729.jpg'),(512,512))

    cv2.imshow('img',img)    

    # main process
    cv2.setMouseCallback('img',rg.mouse_event)


    cv2.waitKey(-1)
    cv2.destroyAllWindows()