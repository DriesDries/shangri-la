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

    Get seed
        - 角度変えてもバイアスつけれるように？
        - 角度変えたフィルタを使う
        - adptive thresholdを使う
        - 種を選ぶための領域選択
            - 重み付け変えてもいい
            - 極大値選ぶ
            - 大きいのを優先的に選ぶ 　
        - parameter tuning
            - threshold
            - sigma 3 ~ 5
            - bias 0 ~ 0.2

    
        - 種をがボールフィルタが最も高いピクセルに置く
        - 大きいの優先でいく
        - 勾配も足す
    
    Region Growing
        - texton map
            - 初めの対象としては、原点の隣のピクセルを見た方が良い
            - 拡張条件に入れるときは、clusterの距離で比較する
            - もしくは、近傍領域まで考える
            - 最大のクラスタにはいかないようにする
        - 平均を見る
        - エッジ強調化とかする？
        - tmの値が高いと、滲み出ることはない
            - tmの値に応じてガウシアン分布を変えるのが良いのでは？
            - 反応したフィルタに応じてスケールを変えて、tmの値に応じて分散を変える？みたいな
        - texture使った方が抑えてはいるっぽい

        - cluster間の距離を見た実装。
        - 近傍領域を見た実装。
        - 大きさに関する評価を行う実装。
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
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import stats

import texton_map as texton

true_ror = cv2.imread('../../../data/g-t_data/label/spirit118-1.png',0)

def main(img, param):
    if param == None:
        param = [170, 4, 0, np.pi, 0, 0, 0]
    seed_img = rd.rd_main(img, param)

    return seed_img


class RockDetection():

    def __init__(self): # インスタンス変数
        
        self.min_size = 10        # 最小領域の大きさ  
        # self.sun_direction = np.pi
        # self.thresh = 160
        self.scale = 10 # いくつのカーネルを用いるか

    def rd_main(self, img, param):

        # Set parameter
        thresh = param[0]
        sigma = param[1]
        bias = param[2]
        direction = param[3]
        
        ## Seed Acquisition by Viola-Jones
        cvmaps = cv.convolution(img.copy(), direction, self.scale, sigma, bias)
        seed_img, seed_list, scale_img = cv.get_seed(img,cvmaps, thresh) # listはy,x
        # new_seed_img, new_seed_list = select_seed(img.copy(),seed_img, scale_img)



        ## Generate Texton map # responsesはmaxの反応したやつ8個
        # texton_map, dis, responses = texton.main(img,radius=6)
        # texton_map[new_seed_list[:,2],new_seed_list[:,3]] = 0
        # plt.imshow(texton_map)
        # plt.pause(1)

        
        ## image clustering by Region Growing and texture analysis
        # ror = rg.main(img.copy(), new_seed_list, scale_img, self.scale, responses, texton_map, dis, param)
        
        # print np.count_nonzero(ror), np.count_nonzero(true_ror), 1. * np.count_nonzero(ror)/np.count_nonzero(true_ror)

        # display result ##########################################################
        # for i in range(1,10):
        # b,g,r = cv2.split(img)
        # b,g,r = cv2.split((img*0.8).astype(np.uint8))
        # r[ror == 255] = 255
        # g[ror == 200] = 255
        # b[seed_img != 0] = 255
        # res = cv2.merge((b,g,r))
        # cv2.imshow('res',res)
        # cv2.waitKey(-1)
        ############################################################################

        return seed_img

class RegionGrowing():

    def main(self, img, seed_list, scale_img, scale, responses, texton_map, dis, param):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
                   scale        -> 画像形式
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
        '''        
        cimg = img.copy()

        # for i in range(5):
            # img = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
            # cv2.imshow('img',img)
            # cv2.waitKey(-1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        rors = np.zeros_like(scale_img).astype(np.uint8)
        gauimgs = self.get_gau_image(scale) # 0-1に正規化,float64 # 複数スケールのgaussian imageの用意



        # for chi in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]:
          # print chi
        rors = np.zeros_like(scale_img).astype(np.uint8)
        for i, seed in enumerate(seed_list):

            if rors[seed[0],seed[1]] == 0: # 新たな種
                
                # 明るい領域
                ror = self.growing(img.copy(), seed, gauimgs[scale_img[seed[0],seed[1]] - 1], responses, texton_map, dis, param) 
                rors[ror==255] = 255
                rors[ror==200] = 200

                # print '{}/{}: y={}, x={}, size={}'.format((i+1),len(seed_list[:,0]),seed[0],seed[1],np.count_nonzero(ror))
                # if np.count_nonzero(ror)>10:
                    # plt.scatter(chi1, chi2, color = 'r')

        # x = range(0,100)
        # plt.plot(x,x)
        # # plt.xlim(0,5)
        # # plt.ylim(0,5)
        # plt.xlabel('Rock -> Rock')
        # plt.ylabel('Rock -> Soil')
        # plt.show()
        # b,g,r = cv2.split(cimg)
        # r[rors!=0] = 255
        # g[seed_list[:,2],seed_list[:,3]] = 0
        # res = cv2.merge((b,g,r))
        # cv2.imshow('res',res)
        # cv2.waitKey(1)

        return rors

    def growing(self, img, seed, gauimg, responses, texton_map, dis, param):
        '''
            args : src          -> img,3ch
                   ori_seed     -> x,yの順の配列
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''

        dif = param[4]
        dif2 = param[5]

        # 準備
        ror = np.zeros_like(img).astype(np.uint8)
        ror[seed[0], seed[1]] = 255
        ror[seed[2], seed[3]] = 255
        ror[seed[4], seed[5]] = 255

        sy = seed[0]
        sx = seed[1]
        seeds = []
        seeds.append([seed[2],seed[3]]) # maxを種に
        # seeds.append([seed[4],seed[5]]) # minを種に

        chi1 = 0
        chi2 = 0
        count1 = 0
        count2 = 0

        # region growing
        for i in xrange(100000):

            if i == len(seeds): # 終了条件
                break
            
            if i >= 3000: # 中断条件
                break

            y,x = seeds[i] # renew seed point

            # 周囲のピクセルと比較
            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):
              
                ## 中断条件 : 画像の端か検出済みだったら
                if u < 0 or u >= img.shape[1] or v < 0 or v >= img.shape[0] or ror[v,u] != 0:
                    continue

                else: ## 継続する
                    ## Calculate gaussian value
                    if abs(v - sy) > int(gauimg.shape[0]/2 -1) or abs(u - sx) > int(gauimg.shape[1]/2 -1):
                        gau = 1.0                
                    else:
                        gau = gauimg[int(v-sy+gauimg.shape[0]/2), int(u-sx+gauimg.shape[1]/2)]

                    ## 領域拡張条件 v,u -> 拡張先 y,x -> 今いるseed seed[2],seed[3] -> もともとのseed
                    E1 = abs(img[v,u] - img[tuple(seeds[0])])
                    # chi = stats.chisquare(responses[:,v,u], responses[:,seed[2],seed[3]])[0]
                    chi = np.linalg.norm(responses[:,v,u] - responses[:, y, x], ord=None)
                    E2 = gau * abs(img[v,u] - img[tuple(seeds[0])])
                    # E3 = img[v,u] < 60
                    # E4 = abs(stats.chisquare(responses[:,v,u], responses[:,y,x])[0])
                    # E5 = dis[texton_map[v,u], texton_map[y,x]] # cluster間の距離
                    # E6 = img[v,u] < img[y,x]
                    # E7 = E1 * E4
                    # if dif > E2 or E3 and i<50:
                    # if dif > E2 or E3:

                    # 岩の名領域
                    # if chi > dif: # 小さいとき
                    if chi > 0.2:
                    # if E2 < dif and chi > dif2:
                        ror[v,u] = 255
                        seeds.append([v,u])
                        # chi1 += chi
                        # count1 += 1
                    
                    # 岩じゃない
                    # else:
                        # chi2 += chi
                        # count2 += 1

                    # elif E2 < 40 and E5 == 0 and i>=100 : # 大きいとき
                    # elif E2 < 30  and i>=100 : # 大きいとき

                        # ror[v,u] = 200
                        # seeds.append([v,u])

        ## dilateとerodeをする
        ror = cv2.dilate(ror,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        ror = cv2.erode(ror,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        # if count1 != 0 and count2 != 0:
            # chi1 = 1.*chi1 / count1
            # chi2 = 1.*chi2 / count2
            # print 'r -> r', chi1
            # print 's -> s', chi2

            # if chi1 < chi2:
                # print 'rock!!!'
            # elif chi2 < chi1:
                # print 's...'

            # plt.scatter(chi2, count2, color = 'b')


        return ror

    def get_gau_image(self,scale):
        '''
            scaleの配列に基づいた複数のガウシアン関数を用意する
        '''

        kernels = []

        sigma = range(12,2,-1) # 分散の定義

        for i in range(scale):
            size = 40 # kernelのサイズ
            # nsig = sigma[i]  # 分散sigma
            nsig = 8  # 分散sigma
            interval = (2*nsig+1.)/(size)
            x = np.linspace(-nsig-interval/2., nsig+interval/2., size+1) # メッシュの定義

            kern1d = np.diff(st.norm.cdf(x)) # 多分ここでガウシアンにしてる,1次元
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d)) # 二次元にしてる
            kernel_raw = cv2.normalize(kernel_raw, 0, 1, norm_type = cv2.NORM_MINMAX)
            kernel_raw = abs(1-kernel_raw)
            kernels.append(kernel_raw)

        return kernels

class ImageConvolution():

    def convolution(self, img, direction, scale, sigma, bias):
        '''
        入力画像に複数のスケールのカーネルを用いてテンプレートマッチングを行う。
        そもそもfilterlingじゃなくてTMにする？
        src: 
        dst: seed
        scaleの数だけ返す
        psiは位相

        '''
        gimg = copy.deepcopy(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

        # kernelの準備
        sizes = range(5, 2*scale+5, 2) # kernelのsize

        kernels = map(lambda x: cv2.getGaborKernel(ksize = (x,x), sigma = sigma, theta = direction, lambd = x, gamma = 25./x, psi = np.pi * 1/2), sizes)

        for i,kernel in enumerate(kernels):
            
            # 正規化するとき
            kernels[i] = 1. * kernel / np.amax(kernel) # Normalized from -1 to 1

            # bias使うとき
            # l = kernel.shape[0]/2
            # kernel[:,0:l] = 2. * kernel[:,0:l] / np.amax(kernel[:,0:l]) - bias
            # kernels[i][:,0:l] = kernel[:,0:l]
            # kernel[:,l+1:] = 2.* kernel[:,l+1:] / abs(np.amin(kernel[:,l+1:])) + bias
            # kernels[i][:,l+1:] = kernel[:,l+1:]
            # print 1+np.argmax(kernel)%sizes[i]
            # kernel = kernel - np.min(kernel)
            # kernel = 255. * kernel / np.max(kernel)
            # print np.max(kernel),np.min(kernel)
            # cv2.imshow('kernel',kernel.astype(np.uint8))
            # cv2.waitKey(-1)


        # filtering # kernelの大きさによる正規化
        vjmaps = map(lambda x:cv2.filter2D(gimg, cv2.CV_64F, x),kernels)
        vjmaps = map(lambda x:vjmaps[x]/(sizes[x]**2),range(len(sizes))) 

        # すべてのvjmapsを通して0-255で正規化
        vjmaps = cv2.normalize(np.array(vjmaps), 0, 255, norm_type = cv2.NORM_MINMAX)
        vjmaps = vjmaps.astype(np.uint8)

        return vjmaps

    def get_seed(self, img, vjmaps, thresh):
        '''
        vjmapsから，種を選択する。
        seed_imgとseed_listとそれぞれのscaleを返す。
        
        seed_img    : seedにvjmapsの値が格納された画像
        seed_list   : seed_imgをlistにしたもの
        seed_list2  : 種が類似度の昇順で並んでるlistスケールが入ってるリスト
        scale       : seed_listに対応したそれぞれのkernelの
        scale_img   : それぞれのピクセルの最も大きいscaleが入ってる
        '''
        
        # 閾値処理
        vjmaps[vjmaps<thresh]=0

        # for vjmap in vjmaps:
        #     b,g,r = cv2.split(img)
        #     r[vjmap!=0] = 255
        #     res = cv2.merge((b,g,r))
        #     cv2.imshow('res',res)
        #     cv2.waitKey(-1)


        # 最大値とそのスケールが入る画像の用意
        maxima = np.zeros_like(vjmaps[0])
        scale_img  = np.zeros_like(vjmaps[0])

        # 各ピクセルの最大値抽出、その値とスケールを画像に入れる
        for i in range(vjmaps.shape[1]):
            for j in range(vjmaps.shape[2]):
                maxima[i,j] = np.max(vjmaps[:,i,j])   # 最大値が入る
                scale_img[i,j]  = np.argmax(vjmaps[:,i,j])+1 # そのときのスケールが入る

        # seedの取得
        # seed_img, scale_img = get_maxima2(maxima, scale_img)
        seed_img = self.get_maxima(maxima) # 各領域の最大値を求める
        # scale_img[seed_img == 0] = 0
        # print np.count_nonzero(seed_img),np.count_nonzero(scale_img)

        # seedの位置をfilterで最も値が高いピクセルにする

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

def select_seed(img, seed_img, scale_img):
    '''
        近傍で最も輝度値が高いピクセルと低いピクセルを種とする
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    new_seed = np.zeros_like(img)
    new_seed_list = []
    
    for j in range(4,seed_img.shape[0]-4):
        for i in range(4,seed_img.shape[1]-4):
            # print j,i
            if seed_img[j,i] != 0:

                ## 最大値と最小値 これをnew_seedに
                ma = np.argmax(img[j-2:j+3,i-2:i+3])
                mi = np.argmin(img[j-2:j+3,i-2:i+3])
                new_seed[j-2 + ma/5, i-2 + ma%5] = 255
                new_seed[j-2 + mi/5, i-2 + mi%5] = 130

                new_seed_list.append([j,i,j-2 + ma/5, i-2 + ma%5,j-2 + mi/5, i-2 + mi%5])

    return new_seed, np.array(new_seed_list)

def get_maxima2(maxima, scale):
    '''
    入力された画像を領域分割し、各領域の最大値を算出する。
    src: 1ch-img
    dst: 領域の最大値のピクセルにのみその値が格納された画像。
    '''

    # 各領域にラベルをつける
    s = maxima.copy()
    s[s!=0] = 255
    
    labels, num = sk.label(s, return_num = True) 
    # print num
    # plt.imshow(s);plt.pause(-1)
    # plt.imshow(labels)
    # plt.pause(-1)

    seed_img = np.zeros_like(maxima)
    
    # 各領域の最大値を求める
    for i in range(1,np.max(labels)):
        ## 初期化
        scale_i = scale.copy() # 初期に戻す
        maxima_i = maxima.copy()

        ## iの領域だけ残す
        scale_i[labels!=i] = 0 # これで残った領域の最大値求める
        maxima_i[labels!=i] = 0 # これで残った領域の最大値求める

        max_scale_i = np.max(scale_i) # 領域の中の最も大きいスケールを種とする
        maxima_i[scale_i != max_scale_i] = 0
        # print np.max(maxima_i)

        seed_img[maxima_i == np.max(maxima_i)] = np.max(maxima_i)

    scale[seed_img == 0] = 0


    return seed_img, scale

'''read class'''
rd = RockDetection()
cv = ImageConvolution()
rg = RegionGrowing()

if __name__ == '__main__':

    # get image
    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    # img = cv2.imread('../../../data/test_image/sample/image14.png')
    # img = cv2.imread('../../../data/g-t_data/resized/spirit006-1.png')


    # main processing

    # for dif2 in np.arange(0.1,0.3,0.03):
      # for dif in np.arange(0,20,3):
        # param = [170, 4, 0, np.pi, dif, dif2]
        # print param
    seed_img = main(img, param=None)

    # cv2.imshow('img',img)
    # plt.show()
    cv2.waitKey(-1)