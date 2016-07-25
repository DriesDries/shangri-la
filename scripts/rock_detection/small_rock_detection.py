# -*- coding: utf-8 -*-
'''
Small Rock Detection based on Viola-Jones and Region Growing Algolithm
Usage: $ python rock_detection.py
argv : Image -> 3ch画像
dst  : Region of Rocks Image -> 1ch image
ラベルとかつけるか？
''' 

import time, math, copy, os, sys
import pdb

import cv2
import numpy as np
import skimage.measure as sk
import skimage.segmentation as skseg
import scipy.stats as st

# import scipy.ndimage as nd
# from scipy import misc
# from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def main(img, direction):
    ror = rd.small_rd(img, direction)
    return ror

class RockDetection():

    def __init__(self): # インスタンス変数
        
        self.min_size = 10        # 最小領域の大きさ  
        # self.sun_direction = np.pi
        self.thresh = 160
        self.scale = 10

    def small_rd(self, src, direction):

        img = copy.deepcopy(src)        

        # Seed Acquisition by Viola-Jones
        vjmaps = vj.vj_main(img, self.scale, direction) 
        seed_img, seed_list, scale = vj.get_seed(vjmaps,self.thresh) # listはy,x
        print 'small seed number = ',len(seed_list)

        # Region Growing Algolithm
        ror, bil_img = rgrow.rg_vj(src, np.array(seed_list), seed_img, scale, self.scale)

        # display result ##########################################################
        # cv2.imshow('src',src)
        
        # cv2.imshow('seed',display_result((img*0.8).astype(np.uint8),seed_img,'fill','r'))
        # cv2.imshow('result', display_result(img, ror, 'fill','r'))
        result = display_result((img*0.8).astype(np.uint8),ror,'fill','r')
        cv2.imshow('result', display_result(result, seed_img, 'fill','g'))

        # cv2.imshow('img',img)
        ############################################################################

        return ror

class ViolaJones():

    def vj_main(self, src, scale, direction):
        '''
        入力画像にtmみたいなことして返す
        src:
        dst: seed
        scaleの数だけ返す
        psiは位相

        '''
        img = copy.deepcopy(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY))
                
        scales = range(5, 5+scale*2, 2)
        kernels= range(len(scales))
        vjmaps = range(len(scales))  

        # kernelの準備
        kernels = map(lambda x:cv2.getGaborKernel(ksize = (x,x), sigma = 5,theta = direction, lambd = x*1, gamma = 25./x, psi = np.pi * 1/2), scales)

        # filtering # kernelの大きさによる正規化
        vjmaps = map(lambda x:cv2.filter2D(img, cv2.CV_64F, x),kernels)
        vjmaps = map(lambda x:vjmaps[x]/(scales[x]**2),range(len(scales))) 

        # すべてのvjmapsを通して0-255で正規化
        vjmaps = cv2.normalize(np.array(vjmaps), 0, 255, norm_type = cv2.NORM_MINMAX)
        vjmaps = vjmaps.astype(np.uint8)

        return vjmaps

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
        return vj_list, value

    def get_seed(self, vjmaps, thresh):
        '''
        vjmapsから，種を選択する
        seed_imgとseed_listとそれぞれのscaleを返す
        seed_img:vjmapsの中のrgのseedだけが255になった画像
        seed_list:seedの中で，昇順に並んだ座標が入ってるリスト
        scale:seed_listに対応したそれぞれのkernelのスケールが入ってるリスト
        '''
        
        # 閾値処理
        vjmaps[vjmaps<thresh]=0
        maxima = np.zeros_like(vjmaps[0])
        scale  = np.zeros_like(vjmaps[0])

        # 各座標の最大値抽出、その値とスケールを画像に入れる
        for i in range(vjmaps.shape[1]):
            for j in range(vjmaps.shape[2]):
                maxima[i,j] = np.amax(vjmaps[:,i,j])   # 最大値が入る
                scale[i,j]  = np.argmax(vjmaps[:,i,j]) # そのときのスケールが入る

        # 各領域の最大値を求める，これが種になる
        # seed_img, _ = detect_maxima(maxima, 0) # sano_algo,極大値
        seed_img = self.get_maxima(maxima) #各領域の最大値を求める

        # この処理をするとseedにvjmapsの値が入る
        # maxima[seed_img!=255] = 0
        # seed_img = maxima

        # imgをlistにする
        seed_list, value = self.img2list(seed_img)

        # valueのorderの取得
        value = np.array(value)
        order = np.argsort(value)[::-1]

        # listをorderに沿った昇順にする
        seed_list2 = []
        for i in range(len(seed_list)):
            seed_list2.append(seed_list[order[i]])

        '''
        seed_img:threshよりも大きい領域のさらに極大値
        scale:それぞれの種が本来どのscaleのkernelのものなのかが格納されてる画像
        seed_list2:種が類似度の昇順で並んでるlist
        '''

        return seed_img, seed_list2, scale

    def get_maxima(self, src):
        '''
        入力された画像から種を求める
        '''

        img = copy.deepcopy(src)
        img[img!=0] = 255
        labels, num = sk.label(img, return_num = True) 

        seed_list = []
        seed_img = np.zeros_like(src)
        
        # 各領域の最大値を求める
        for i in range(1,num+1):
            
            # iの領域だけ残す
            img = copy.deepcopy(src) # 初期に戻す
            img[labels!=i] = 0 # これで残った領域の最大値求める
            
            # 最大値を求める
            y = np.argmax(img)/len(img)
            x = np.argmax(img)%len(img)

            if img[y,x] != 0: # 中に空いた穴じゃなければ種にする
                seed_img[y,x] = src[y,x]
                # seed_list.append([y,x])
        
        return seed_img

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

        rors = np.zeros_like(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)).astype(np.uint8)

        img = copy.deepcopy(src)
        # img = self.preprocessing(img) # エッジがよく強調されない
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16)

        # フィルタごとのgaussian imageの用意
        gauimgs = em.gau_function(scale_number) # 0-1に正規化,float64

        for i, y, x in zip( range(len(seed_list[:,0])) , seed_list[:,0], seed_list[:,1]):
            print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y            
            if rors[y,x] == 0: # 新たな種
                ror = self.growing_vj(img, x, y, gauimgs[scale[y,x]]) 
                rors[ror==255] = 255 # rener rors
                # cv2.imshow('growing',rors)
                # cv2.waitKey(0)

        return rors, img

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
                E2 = gau * abs(shade_ave - img[v,u])
                if 10 > E1 or 10 > E2:
                
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

    def preprocessing(self,src):
        '''
            領域拡張する前処理
        cv2.bilateralFilter
        d(diameter)-> 処理時に考慮するピクセルの数(直径)
        sigmacolor -> 色がより均一になる
        sigmaspace -> より広範囲のpixelまで効果が及ぶ

        '''
        img = copy.deepcopy(src)

        for i in range(5):
            print i 
            img = cv2.bilateralFilter(img, d=5, sigmaColor=15, sigmaSpace=15)
            cv2.imshow('img',img)
            cv2.waitKey(0)

        return img



class EnergyMinimization():

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

    def em_vj(self,src,seed_list,seed_img):
        
        dd = DisplayData()
        # フィルタの大きさは7とする
        # もっとも反応するフィルタは、岩よりも大きいはず
        # 11でやってる
        img = copy.deepcopy(src)
        img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        img2= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY) 
        
        seed = seed_list[80]

        x = seed[0]
        y = seed[1]
        print x,y

        img = img[y-10:y+10,x-10:x+10]
        img[10,10] = 0

        # 平均の準備,明，暗それぞれ, あと中間も定義する，そしてrg
        # もしくは明と暗をつなぐように中間？
        light_ave = (img2[y-1,x-1]*1. + img2[y,x-1] + img2[y+1,x-1])/3 # これだとfloatになる
        shade_ave = (img2[y-1,x+1]*1. + img2[y,x+1] + img2[y+1,x+1])/3 # これだとfloatになる
        # print light_ave,shade_ave

        ''' ガウシアン関数の用意 '''
        kernlen = 20
        nsig = 3
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = cv2.normalize(kernel, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
        kernel = 255 - kernel
        kernel = kernel.astype(np.float)
        kernel = cv2.normalize(kernel, 0, 1, norm_type = cv2.NORM_MINMAX)

        # 平均と分散の用意、kernelの大きさが11だから、それに基づいて確実に岩っぽいとことる

        E = np.zeros_like(img)

        # エネルギー導出
        for i in range(20):
            for j in range(20):
                print i,j
                E[i,j] =  abs(img[i,j]-light_ave)*kernel[i,j]

        # 試しに見てみる
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        b,g,r = cv2.split(img)
        r0 = copy.deepcopy(r)
        g0 = copy.deepcopy(g)
        for i in range(1,100,10):
            print i
            r = r0
            g = g0
            r[E<i] = 255
            g[E<i] = 0
            img = cv2.merge((b,g,r))
            cv2.imshow('img',img)
            cv2.waitKey(0)
            
        return 0,0

def display_result(img,mask,format,color):
        '''
        imgの上にmaskを重ねて表示する
        img : 3ch, 512x512
        mask: 1ch, 512x512
        format: str, fill or edge
        '''
        # print type(img.dtype)
        if len(img.shape) == 2:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        # colorだったらgrayに変換
        if len(mask.shape) == 3:
            img = img.astype(np.uint8)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

        # dtypeがuint8じゃなければ変換
        if mask.dtype != 'uint8':
            mask = cv2.normalize(mask, 0, 255, norm_type = cv2.NORM_MINMAX)
            mask = mask.astype(np.uint8)

        ''' fill '''
        b,g,r = cv2.split(img)
        if color == 'r':
            r[mask != 0] = 255
            # g[mask != 0] = 0
            # b[mask != 0] = 0

        elif color == 'g':
            g[mask != 0] = 255
            r[mask != 0] = 0
            b[mask != 0] = 0

        else:
            b[mask != 0] = 255
        fill_result = cv2.merge((b,g,r))

        # エッジにする
        mask = cv2.Canny(mask, 0, 0,apertureSize = 3)
        b,g,r = cv2.split(img)
        if color == 'r':
            r[mask != 0] = 255
            # g[mask != 0] = 0
            # b[mask != 0] = 0        

        elif color == 'g':
            r[mask != 0] = 0
            g[mask != 0] = 255
            b[mask != 0] = 0  

        else:
            r[mask != 0] = 0
            g[mask != 0] = 0
            b[mask != 0] = 255              

        edge_result = cv2.merge((b,g,r))


        if format == 'fill':
            result = fill_result
        elif format == 'edge':
            result = edge_result
        else:
            result = img


        return result

def detect_maxima(src,thresh):
        '''
        閾値よりも大きい極大値を見つけて，imgとlistとして返す
        '''

        # colorだったらgrayに変換
        if len(src.shape) == 3:
            src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        src_gray=np.asarray(copy.deepcopy(src))

        src_gray[src_gray<thresh] = 0 # 閾値処理

        gray_shifted = src_gray[0:src_gray.shape[0],0:-1:]
        zero_comp_ver = np.zeros((src_gray.shape[0],1))
        zero_comp_ver = zero_comp_ver.astype(np.int16)
        right_img= np.append(zero_comp_ver,gray_shifted,1)
        right_img= np.asarray(right_img)

        gray_shifted = src_gray[0:src_gray.shape[0],1::]
        left_img= np.append(gray_shifted,zero_comp_ver,1)
        left_img = np.asarray(left_img)

        gray_shifted = src_gray[1:src_gray.shape[0],0::]
        zero_comp_holi = np.zeros((1,src_gray.shape[1]))
        zero_comp_holi = zero_comp_holi.astype(np.int16)
        up_img = np.append(gray_shifted,zero_comp_holi,0)
        up_img = np.asarray(up_img)

        gray_shifted = src_gray[0:src_gray.shape[0]-1,0::]
        down_img = np.append(zero_comp_holi,gray_shifted,0)
        down_img = np.asarray(down_img)

        right_img = right_img.astype(np.int16)
        left_img = left_img.astype(np.int16)
        up_img = up_img.astype(np.int16)
        down_img = down_img.astype(np.int16)
        src_gray = src_gray.astype(np.int16)
        '''
        グレー元画像とシフト画像との差分
        '''
        sub_right_img=src_gray-right_img
        sub_left_img=src_gray-left_img
        sub_up_img=src_gray-up_img
        sub_down_img=src_gray-down_img

        k=0
        #Map=np.zeros((src_gray.shape[0]*src_gray.shape[1],2))
        '''
        返り値の用意
        '''
        Map=[]
        #print (sub_right_img.dtype)
        maps = np.zeros_like(src_gray)
        #print (maps.dtype)
        '''
        極値の探索
        line88,105,123,140のレンジの第２要素目の+1の必要性を吟味
        '''

        for i in range(0,src_gray.shape[0]):
            for j in range(0,src_gray.shape[1]):

                # 上下左右のすべてより大きい場合
                if sub_right_img[i,j]>0 and sub_left_img[i,j]>0 and sub_up_img[i,j]>0 and sub_down_img[i,j]>0:
                    Map.append([j,i])
                    maps[i,j] = 255
                    k=k+1

                # 横に続く場合
                elif(sub_right_img[i,j]==0) and (sub_left_img[i,j]>0) and (sub_up_img[i,j]>0) and (sub_down_img[i,j]>0) :
                    l=j-1

                    while (l>0):

                        if(sub_right_img[i,l]<0 or sub_up_img[i,l]<=0 or sub_down_img[i,l]<=0):

                            break
                        elif (sub_right_img[i,l]>0):
                            for r in range(l,j+1):
                                Map.append([i,r])
                                maps[r,i]=255
                                k=k+1
                            break
                        elif sub_right_img[i,l]==0:
                            l = l-1

                        else:
                            break

                # # 縦に続く場合
                elif sub_right_img[i,j]>0 and sub_left_img[i,j]>0 and sub_up_img[i,j]==0 and sub_down_img[i,j]>0:
                    l=i+1

                    while (l<src_gray.shape[0]):
                        if(sub_up_img[l,j]<0 or sub_right_img[l,j]<=0 or sub_left_img[l,j]<=0):
                            break
                        elif (sub_up_img[l,j]>0):
                            for r in range(i,l+1):
                                Map.append([j,r])
                                maps[r,j]=255
                                k=k+1
                            break
                        elif (sub_up_img[l,j]==0):
                            l=l+1
                        else:
                            break
                else:
                    continue

        maps = maps.astype(np.uint8)
        # print 'seed number =',k

        maxima_list = np.array(Map)

        return maps

'''read class'''
rd = RockDetection()
rgrow = RegionGrowing()
vj = ViolaJones()
em = EnergyMinimization()


if __name__ == '__main__':

    # get image
    # img = cv2.resize(cv2.imread('../../../image/rock/spiritsol118navcam.jpg'),(512,512))
    # img = cv2.resize(cv2.imread('../../../image/rock/sol729.jpg'),(512,512))
    # img = cv2.resize(cv2.imread('../../../image/rock/11.png'),(512,512))
    img = cv2.imread('../../../image/rock/spiritsol118navcam.jpg')
    img = img[400:800,400:800]

    # main processing
    main(img, np.pi)

    cv2.waitKey(-1)