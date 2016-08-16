# -*- coding: utf-8 -*-
'''
Small Rock Detection based on Viola-Jones and Region Growing Algolithm
Viola-Jones法に基づいて複数のスケールのカーネルによるテンプレートマッチングを行うことで、岩領域の抽出を行う。
TMの結果に対して閾値処理を行い、その各領域から最大値とそのときに用いたテンプレートを求める。
それらのピクセルを領域拡張法の種として、その種を中心としてテンプレートの大きさに基づいたガウス分布を展開し、
エネルギー関数を導入し、定めたエネルギーの閾値よりも小さい場合は、隣のピクセルと結合する。
エネルギー関数 E:
用いたテンプレートのパラメータ:


Usage: $ python rock_detection.py
argv : Image -> 3ch画像
dst  : Region of Rocks Image -> 1ch image

・テンプレートマッチングにする
    - スコアの正規化がいらない
・カーネル最適化
    - 大きさ、カーネルそのものを変更
・類似度指標の変更
・種の取得きれいに
・gauimgの最適化
・パラメータの最適化

''' 

import time
import math
import copy
import os
import sys

import cv2
import numpy as np
import skimage.measure as sk
import skimage.segmentation as skseg
import scipy.stats as st

def main(img, direction):
    
    ror = rd.small_rd(img, direction)

    return ror

class RockDetection():

    def __init__(self): # インスタンス変数
        
        self.min_size = 10        # 最小領域の大きさ  
        # self.sun_direction = np.pi
        self.thresh = 160
        self.scale = 10 # いくつのカーネルを用いるか

    def small_rd(self, src, direction):


        img = copy.deepcopy(src)        

        # Seed Acquisition by Viola-Jones
        vjmaps = vj.vj_main(img, direction, self.scale) 
        seed_img, seed_list, scale_img = vj.get_seed(vjmaps,self.thresh) # listはy,x
        
        # Region Growing Algolithm
        ror = rgrow.rg_vj(src, seed_list, seed_img, scale_img, self.scale)

        # display result ##########################################################
        # print 'small seed number = ',len(seed_list)

        ############################################################################

        return ror

class ViolaJones():

    def vj_main(self, src, direction, scale):
        '''
        入力画像に複数のスケールのカーネルを用いてテンプレートマッチングを行う。
        そもそもfilterlingじゃなくてTMにする？
        src: 
        dst: seed
        scaleの数だけ返す
        psiは位相

        '''
        img = copy.deepcopy(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY))

        # カーネルの準備                
        sizes = range(5, 2*scale+5, 2) # kernelのsize

        # kernelの準備
        kernels = map(lambda x: cv2.getGaborKernel(ksize = (x,x), sigma = 5,theta = direction, lambd = x, gamma = 25./x, psi = np.pi * 1/2), sizes)

        # filtering # kernelの大きさによる正規化
        vjmaps = map(lambda x:cv2.filter2D(img, cv2.CV_64F, x),kernels)
        vjmaps = map(lambda x:vjmaps[x]/(sizes[x]**2),range(len(sizes))) 

        # すべてのvjmapsを通して0-255で正規化
        vjmaps = cv2.normalize(np.array(vjmaps), 0, 255, norm_type = cv2.NORM_MINMAX)
        vjmaps = vjmaps.astype(np.uint8)

        return vjmaps

    def get_seed(self, vjmaps, thresh):
        '''
        vjmapsから，種を選択する。
        seed_imgとseed_listとそれぞれのscaleを返す。
        
        seed_img    : seedにvjmapsの値が格納された画像
        seed_list   : seed_imgをlistにしたもの
        seed_list2  : 種が類似度の昇順で並んでるlistスケールが入ってるリスト
        scale       : seed_listに対応したそれぞれのkernelの
        '''
        
        # 閾値処理
        vjmaps[vjmaps<thresh]=0
        
        # 最大値とそのスケールが入る画像の用意
        maxima = np.zeros_like(vjmaps[0])
        scale_img  = np.zeros_like(vjmaps[0])

        # 各座標の最大値抽出、その値とスケールを画像に入れる
        for i in range(vjmaps.shape[1]):
            for j in range(vjmaps.shape[2]):
                maxima[i,j] = np.amax(vjmaps[:,i,j])   # 最大値が入る
                scale_img[i,j]  = np.argmax(vjmaps[:,i,j]) # そのときのスケールが入る

        # seedの取得
        seed_img = self.get_maxima(maxima) # 各領域の最大値を求める
        seed_list, value = self.img2list(seed_img)

        # 昇順の要素番号の取得
        order = np.argsort(value)[::-1]
        
        # listをorderに沿った昇順にする
        seed_list2 = []
        for i in range(len(seed_list)):
            seed_list2.append(seed_list[order[i]])

        return seed_img, np.array(seed_list2), scale_img

    def get_maxima(self, src):
        '''
        入力された画像を領域分割し、各領域の最大値を算出する。
        src: 1ch-img
        dst: 領域の最大値のピクセルにのみその値が格納された画像。
        '''

        img = copy.deepcopy(src)
        img[img!=0] = 255

        # 各領域にラベルをつける
        labels, num = sk.label(img, return_num = True) 

        seed_list = []
        seed_img = np.zeros_like(src)
        
        # 各領域の最大値を求める
        for i in range(1,num+1):

            # iの領域だけ残す
            img = copy.deepcopy(src) # 初期に戻す
            img[labels!=i] = 0 # これで残った領域の最大値求める
            
            # 最大値を求める,1行にしたときの値が出てくるからこんなになってる
            y = np.argmax(img)/len(img)
            x = np.argmax(img)%len(img)

            if img[y,x] != 0: # 中に空いた穴じゃなければ種にする
                seed_img[y,x] = src[y,x]
                # seed_list.append([y,x])
        
        return seed_img

    def img2list(self,img):
        ''' 
        画像で非0の座標をlistに

        '''
        vj_list = []
        # vj_list = np.empty((0,2),np.int16)
        value = []

        for i in range(len(img)):
            for j in range(len(img)):
                if img[j,i] != 0:
                    vj_list.append([j,i])
                    value.append(img[j,i])
        
        return vj_list, np.array(value)

class RegionGrowing():
    
    def __init__(self):
        
        # パラメータ
        self.thresh_rg = 1     # 結合条件の閾値
        self.thresh_sm = 20
        self.region_number = 20000
        self.max_size = 1000

        self.ave_kernel = np.array([[0.11,0.11,0.11],
                                    [0.11,0.11,0.11],
                                    [0.11,0.11,0.11]])

    def rg_vj(self, src, seed_list, seed_img, scale, scale_number):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
                   scale        -> 画像形式
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
        '''        

        img = copy.deepcopy(src)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16)
        rors = np.zeros_like(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)).astype(np.uint8)

        # フィルタごとのgaussian imageの用意
        gauimgs = self.gau_function(scale_number) # 0-1に正規化,float64

        for s in range(10):
         for i, y, x in zip( range(len(seed_list[:,0])) , seed_list[:,0], seed_list[:,1]):
            # print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y            
            if rors[y,x] == 0: # 新たな種
                ror = self.growing_vj(img, x, y, gauimgs[scale[y,x]]) 
                rors[ror==255] = 255 # renew rors
         
        return rors

    def growing_vj(self, src, ori_x, ori_y, gauimg):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   ori_seed     -> x,yの順の配列
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''
        img = copy.deepcopy(src)
        sx = copy.deepcopy(ori_x)
        sy = copy.deepcopy(ori_y)

        # 準備
        region_map = np.zeros_like(img).astype(np.uint8)
        region_map[ori_y,ori_x] = 255
        seed = []
        seed.append([ori_x,ori_y])

        # seedの周囲のピクセルの平均値の算出
        if sx == 0 or sy == 0 or sx >= img.shape[0]-1 or sy >=img.shape[1] - 1: 
            light_ave = 0
            shade_ave = 0
        else:
            light_ave = (src[sy-1,sx-1]*1. + src[sy,sx-1] + src[sy+1,sx-1])/3
            # shade_ave = (src[sy-1,sx+1]*1. + src[sy,sx+1] + src[sy+1,sx+1])/3
            shade_ave = float(min(src[sy-1,sx+1]*1., src[sy,sx+1], src[sy+1,sx+1]))

        # region growing
        for i in xrange(100000):

            if light_ave == 0 and shade_ave == 0:
                break
            if i == len(seed):
                break
            
            # renew seed point
            x,y = seed[i]
            # distance = math.sqrt( (y-sy)**2 + (x-sx)**2 )
            
            # 1pixelごとの処理
            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):

                # gau_kernelより外だとgau=1にする
                if abs(v - sy) > int(gauimg.shape[0]/2 -1) or abs(u - sx) > int(gauimg.shape[1]/2 -1):
                    gau = 1.0                
                else:
                    gau = gauimg[int(v-sy+gauimg.shape[0]/2), int(u-sx+gauimg.shape[1]/2)]
             
                ''' 領域拡張条件 '''
                E1 = gau * abs(light_ave - img[v,u])
                # E2 = gau * abs(shade_ave - img[v,u])
                # if 10 > E1 or 10 > E2:
                if 10 > E1 :
                
                    # renew region map
                    if region_map[v,u] == 0: # 新しい種だった場合
                        region_map[v,u] = 255
                        seed.append([u,v])
                         # count += 1

                # 処理中断条件
                if u == 0 or u >= len(src[0])-1 or v == 0 or v >= len(src[:,0])-1:
                    break
                else:
                    continue

        return region_map

    def gau_function(self,scale):
        '''
            scaleの配列に基づいた複数のガウシアン関数を用意する
        '''

        kernels = []

        sigma = range(12,2,-1) # 分散の定義

        for i in range(scale):
            size = 40 # kernelのサイズ
            nsig = sigma[i]  # 分散sigma
            interval = (2*nsig+1.)/(size) # 
            x = np.linspace(-nsig-interval/2., nsig+interval/2., size+1) # メッシュの定義

            kern1d = np.diff(st.norm.cdf(x)) # 多分ここでガウシアンにしてる,1次元
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d)) # 二次元にしてる
            kernel_raw = cv2.normalize(kernel_raw, 0, 1, norm_type = cv2.NORM_MINMAX)
            kernel_raw = abs(1-kernel_raw)
            kernels.append(kernel_raw)
            # cv2.imshow('kernels',kernels[i])
            # cv2.waitKey(0)

        return kernels

'''read class'''
rd = RockDetection()
rgrow = RegionGrowing()
vj = ViolaJones()

if __name__ == '__main__':

    # get image
    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')

    # main processing
    main(img, np.pi)

    cv2.waitKey(-1)