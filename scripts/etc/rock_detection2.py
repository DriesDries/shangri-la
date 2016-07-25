# -*- coding: utf-8 -*-
'''
Rock Detection based on Saliency Map and Region Growing Algolithm
Usage: $ python rock_detection.py
argv : Image
dst  : Region of Rocks, 1ch
''' 

import time
import math
import copy
# import os
# import sys

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

class RockDetection():
    
    def __init__(self):
        
        self.min_size = 10        # 最小領域の大きさ  
        
        #　paramater for sm
        self.thresh_sm_maxima = 150   # 極大値求めるときのsmの閾値
        self.kernel_size = (11,11)  # dilateとerodeのkernelの大きさ
        
        # paramater for vj
        self.sun_direction = np.pi
        self.vj_thresh_value = 180

    def rock_detection(self,src):
        '''
            岩領域検知
            args : img     -> img,3ch
            dst  : ror_img -> 1chの二値画像, !=0が岩
                   num     -> 岩領域の数
            param:      -> 
            
            ---algolithm---
            まず、元画像からSaliencyMapを生成し、生成したSMから閾値以上の全極大値を求め、その極大値をseedとして保存する。そして、そのseedから領域拡張法に基づいて領域分割を行うことで、岩領域を算出する。
        '''
        img = copy.deepcopy(src)
        
        # get region of rock
        start = time.time()
        sm_seed,sm_ror = self.sm_rd(img)
        print 'sm_ror acquisition'
        vj_seed,vj_ror = self.vj_rd(img)
        print 'vj_ror acquisition'
        print 'processing time = ',time.time() - start
        
        # 小さい領域の削除
        sm_ror, num, bounding_box = self.del_small(sm_ror, self.min_size)
        # vj_ror, num, bounding_box = self.del_small(vj_ror, self.min_size)
        

        sm_ror[vj_ror!=0]=255
        cv2.imshow('ror',sm_ror)


        # 結果を重ねる
        # result = np.zeros_like(src).astype(np.uint8)
        # b,g,r = cv2.split(result)
        # b2,g2,r2 = cv2.split((src*0.8).astype(np.uint8))
        
        # r[sm_ror !=0] = 255
        # g[vj_ror !=0] = 255
        # g[g == r] = 0
        
        # r_edge = cv2.Canny(r, 0, 0,apertureSize = 3)
        # g_edge = cv2.Canny(g, 0, 0,apertureSize = 3)
        # r2[r_edge>200] = 255
        # g2[g_edge>200] = 255
        
        # result = cv2.merge((b,g,r))
        # result2  = cv2.merge((b2,g2,r2))
        
        # result = self.display_result(src,sm_ror,'edge','r')

        '''display results'''###########################################################
        # input and output
        cv2.imshow('in',src)
        # cv2.imshow('Input', self.display_result(src,sm_seed,'fill','r'))
        # cv2.imshow('s_l_result',result)
        # cv2.imshow('edge_result',result2)
        # cv2.imshow('fill_result',self.display_result(src,result,'fill','r'))
        
        # seed image
        # cv2.imshow('sm_seed', self.display_result((src*0.8).astype(np.uint8),sm_seed,'fill','r'))
        cv2.imshow('vj_seed', self.display_result((src*0.8).astype(np.uint8),vj_seed,'fill','r'))

        # ror image
        # cv2.imshow('sm_ror', self.display_result(result,sm_seed,'fill','g'))

        # cv2.imshow('vj_ror', self.display_result((src*0.8).astype(np.uint8),vj_ror,'edge','r'))
        ################################################################################

    def sm_rd(self,img):

        ''' Read class '''
        src = copy.deepcopy(img)
        smap = SaliencyMap()
        rgrow = RegionGrowing()

        '''Seed Acquisition by Saliency Map'''
        # get saliency map
        sm = smap.SaliencyMap(img) # 2をつけるかつかないか
        ori_sm = copy.deepcopy(sm)
        # get seeds listはy,x(画像中で)の多行二列
        # seed_img,seed_list = self.get_maxima(sm,self.thresh_sm_maxima) # scipy-algo
        sm_seed_img, sm_seed_list = self.detect_maxima(sm,self.thresh_sm_maxima) # sano-algo
        print 'sm seed number = ',len(sm_seed_list)

        '''Region Growing'''
        # saliency mapをdilation and erosion
        # sm = cv2.dilate(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size))
        # sm = cv2.erode(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size))
        
        # smの正規化, 値が小さいとhigh saliency
        sm_inv = (255 - sm).astype(np.float32)
        sm_inv = cv2.normalize(sm_inv, 0, 1, norm_type = cv2.NORM_MINMAX)

        # Region Growing　Algorithm
        sm_ror, bil_img = rgrow.rg_sm(img,sm_inv,sm_seed_list)

        '''display results'''###########################################################
        # cv2.imshow('Original Saliency Map',self.display_result(ori_sm,sm_seed_img,'fill','r'))
        # cv2.imshow('Bilateral Image',self.display_result(bil_img,sm_seed_img,'fill','g'))
        # cv2.imshow('sm_ror_bil',self.display_result(bil_img,sm_ror,'fill','r'))

        # cv2.imshow('d_e Saliency Map', sm)        

        # cv2.imshow('sm_seed_img',self.display_result((o*0.8).astype(np.uint8),sm_seed_img,'fill','r'))
        # cv2.imshow('sm_ror',self.display_result((src*0.8).astype(np.uint8),sm_ror,'edge','r'))
        ################################################################################

        return sm_seed_img,sm_ror

    def vj_rd(self,src):

        ''' Read class '''
        img = copy.deepcopy(src)
        rgrow = RegionGrowing()
        vj = ViolaJones()

        cv2.rectangle(img,(220,310),(245,330),(255,255,255))

        '''Seed Acquisition by Viola-Jones'''
        # Filtering
        vjmaps, kernel3,kernel5,kernel7 = vj.vj_main(img,self.sun_direction) 
        # 複数のkernelとそのときのmap # r,g,bの順でkernelが大きい
        # vjmap,vjmap2 = vj.vj_map(vjmaps) # vjmap2はrgbのどれかに入ってる

        vj_seed_list, vj_seed_img = vj.seed(vjmaps[:,:,2],self.vj_thresh_value) # listはy,x
        vj_seed_img, num, bounding_box = self.del_small(vj_seed_img, 2) # 小さいのと端のを削除
        print 'vj seed number = ',len(vj_seed_list)

        '''Region Growing by Viola-Jones'''
        vj_ror, bil_img = rgrow.rg_vj(img,vj_seed_list,vjmaps[:,:,2])

        '''display results'''###########################################################
        # cv2.imshow('vj_seed', self.display_result((img*0.8).astype(np.uint8),vj_seed_img,'fill','r'))

        ################################################################################

        return vj_seed_img,vj_ror

    def get_maxima(self,sm,thresh):
        '''
            入力された画像中の極大値を取得する
            args : img    -> 1ch
            dst  : maxima_img  -> 極大値が255となった二値画像
            param: maxima_list -> 極大値のlist、[y,x]の順で入った多行二列の配列
        '''
        img = copy.deepcopy(sm)

        img[img<thresh] = 0 # 閾値処理
        # img[0:130,0:511] = 0 # 空消す
        # img[0:400,0:1022] = 0 # 空消す

        img90 = np.rot90(img) 

        # 横と縦方向の極大値求める
        maxima   = np.array(signal.argrelmax(img))
        maxima90 = np.array(signal.argrelmax(img90))

        # 回転してたのを元の画像に戻す
        maxima2 = np.empty((2,0),np.int16)
        for i,x,y in zip(range(len(maxima90[0])),maxima90[0],maxima90[1]):            # 
            maxima2 = np.append(maxima2, np.array([[y],[len(img)-1-x]]), axis=1)

        # 縦横ともに極大値の画像の生成
        maxima_img = cv2.cvtColor(np.zeros_like(img),cv2.COLOR_GRAY2BGR)
        maxima_img = maxima_img.astype(np.uint8)
        b,g,r = cv2.split(maxima_img)
        for x,y in zip(maxima[0],maxima[1]):
                r[x,y] = 255
        for x,y in zip(maxima2[0],maxima2[1]):
                g[x,y] = 255                
        maxima_img = cv2.merge((b,g,r))
        maxima_img = cv2.cvtColor(maxima_img,cv2.COLOR_BGR2GRAY)
        _,maxima_img = cv2.threshold(maxima_img , 180, 255,0)
        
        seeds = np.empty((2,0),np.int16)
        for x in range(len(maxima_img)):
            for y in range(len(maxima_img)):
                if maxima_img[x,y] == 255:
                    seeds = np.append(seeds, np.array([[x],[y]]),axis=1)
                else:
                    continue

        maxima_list = np.rot90(seeds)
        return maxima_img, maxima_list

    def detect_maxima(self,src,thresh):
        
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
                    Map.append([i,j])
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
                                maps[i,r]=255
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
                                Map.append([r,j])
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

        return maps,maxima_list

    def del_small(self,img,min_size):
        '''
            args :      -> 画像,1ch
            dst  :      -> 端に繋がってるのと小さいやつ除去した画像
            param:      -> 
        '''

        bounding_box = np.zeros_like(img)

        # 画像の端に繋がってるやつ削除
        cleared = img.copy()
        skseg.clear_border(cleared)

        # ラベリング
        labels, num = sk.label(cleared, return_num = True) # numが一個多い？
        
        # bounding boxの描画
        # for region in sk.regionprops(labels):
            
            # minr, minc, maxr, maxc = region.bbox

            # if region.area < min_size:
                # cleared[minr:maxr,minc:maxc][region.convex_image == True] =0 
                # num = num - 1
                
            # else:    
                # cv2.rectangle(bounding_box,(minc,minr),(maxc,maxr),255)
        bounding_box = 0

        return cleared, num, bounding_box

    def display_result(self,img,mask,format,color):
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

        # 岩領域の色を変える
        b,g,r = cv2.split(img)
        if color == 'r':
            r[mask != 0] = 255
            # g[mask != 0] = 0
            # b[mask != 0] = 0

        elif color == 'g':
            g[mask != 0] = 255
            # r[mask != 0] = 0
            # b[mask != 0] = 0

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

class ViolaJones():

    def vj_main(self,src,sun_direction):
        '''
        入力画像にtmみたいなことして返す
        src:
        dst: seed
        '''
        img = copy.deepcopy(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY))
                
        kernel3,kernel5,kernel7 = map(lambda x:cv2.getGaborKernel(ksize = (x,x), sigma = 5,theta = sun_direction, lambd = x, gamma = 25/x,psi = np.pi * 1/2),[3,5,11])

        vj3,vj5,vj7 = map(lambda x:cv2.filter2D(img, cv2.CV_32F, x),[kernel3,kernel5,kernel7])
        vj3,vj5,vj7 = map(lambda x:cv2.normalize(x, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8),[vj3,vj5,vj7])
        vjmaps = cv2.merge((vj7,vj5,vj3))

        return vjmaps,kernel3,kernel5,kernel7

    def seed(self,img,thresh):

        # 画像をリストに
        vj_list = []
        vj_list = np.empty((0,2),np.int16)
        
        for i in range(len(img)):
            for j in range(len(img)):
                if img[j,i] >= thresh:
                    vj_list = np.append(vj_list, np.array([[j,i]]),axis=0)

        img2 = copy.deepcopy(img)
        img2[img2<thresh] = 0
        img2[img2!=0] = 255

        return vj_list, img2

    def vj_map(self,img):
        '''入力画像のrgbのうち最大のだけを入れた1ch画像を返す'''
        b,g,r = cv2.split(img)
        max_map = np.zeros_like(b)
        max_map2 = copy.deepcopy(img)


        for i in range(len(b[0])):
            for j in range(len(b[1])):

                if b[j,i]>=r[j,i] and b[j,i]>=g[j,i]:
                    # largeが大きい
                    max_map[j,i] = b[j,i]
                    max_map2[j,i,1] = 0
                    max_map2[j,i,2] = 0

                elif g[j,i]>=r[j,i] and g[j,i]>=b[j,i]:
                    # midiumが大きい
                    max_map[j,i] = g[j,i]
                    max_map2[j,i,0] = 0
                    max_map2[j,i,2] = 0

                elif r[j,i]>=b[j,i] and r[j,i]>=g[j,i]:
                    # smallが大きい
                    max_map[j,i] = r[j,i]
                    max_map2[j,i,0] = 0
                    max_map2[j,i,1] = 0

        # cv2.imshow('map',max_map)
        # cv2.imshow('map2',max_map2)

        return max_map,max_map2

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

    def rg_sm(self,src,sm,seed_list):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
            # sigmacolorが大きいと、色がより均一になる,    sigmaspaceが大きいと、より遠くのピクセルまで巻き込むdは処理するときに見るピクセルの直径

        '''
        img = copy.deepcopy(src)        
        # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        
        # 岩領域の画像用意
        rors = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)).astype(np.uint8)

        # bilateralfilterによる処理
        for i in range(5):
            img = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img16 = img.astype(np.int16)
        
        for i,y,x in zip(range(len(seed_list[:,0])),seed_list[:,0],seed_list[:,1]):
            # print y,x
            if rors[y,x] == 0: # 新たな種
            # if rors[y,x] == 0: # 新たな種

                # print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y
                rors[y,x] == 255
                ror = self.growing_sm(img16,sm,x,y) # 新たなRegionの取得
                rors[ror==255] = 255 # Region of Rocksの更新
                # cv2.imshow('growing',rors)
                # cv2.waitKey(0)
                # print i

                # em = EnergyMinimization()
                # 最小化に基づいていらない部分消した領域が帰って来ればok
                # rors2 = em.energy_minimization(src,ror,sm)


            if i == self.region_number:
                break
            else: # すでに描画されたror
                continue
        return rors, img

    def growing_sm(self,src,sm,x,y):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''

        img = copy.deepcopy(src)

        # 準備
        region_map = np.zeros([len(src[0]),len(src[:,0])]).astype(np.uint8)
        region_map[y,x] = 255
        seed = []
        seed.append([x,y])
        count = 1
        sum_val = np.array(img[y,x]).astype(np.int32)
        region = 1

        sy = copy.deepcopy(y)
        sx = copy.deepcopy(x)
        var2 = 0

        # region growing
        for i in xrange(100000):

            # 一つの領域の最大面積、分割が終わったら終了
            # if i == self.max_size:
                # print 'size over'
                # break
            if i == len(seed):
                break
            
            # renew seed point
            x,y = seed[i]

            # 画面の端だったら中断
            if x == 0 or x >= len(src[0])-1 or y == 0 or y >= len(src[:,0])-1:
                break

            # 1pixelごとの処理
            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):

                # 領域拡張の条件式
                # if 5 > abs(img[v,u]-ave):
                # if sm[v,u]<0.7 and ( 20 > abs(img[v,u]-ave) or 3>abs(img[v,u]-img[y,x])):
                
                # エネルギー関数
                # E = sm[v,u] + abs(img[v,u]-ave) + abs(img[v,u]-img[y,x])
                
                if 20 > abs(img[v,u]-img[y,x]) + 35*sm[v,u]:                
                # if E<100:

                    # or 3>abs(img[v,u]-img[y,x])

                    # それまでの岩領域の平均値の計算
                    region = region + 1
                    sum_val = sum_val + img[v,u]
                    ave = sum_val / region
                    # var = 1/region * var2
                    # print img[v,u] - ave, img[v,u] - img[y,x], img[v,u] - img[sy,sx]

                    # renew region map
                    if region_map[v,u] == 0: # 新しい種だった場合
                        region_map[v,u] = 255
                        seed.append([u,v])
                        count += 1

                else:
                    continue
        # print 'seed x,y = ',seed[0],'size =', region, 'ave = ', ave

        return region_map

    def rg_vj(self,src,seed_list,seed_img):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
        '''
        rors = np.zeros_like(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)).astype(np.uint8)

        # 画像の準備,float32にする
        img = copy.deepcopy(src)
        # img = cv2.bilateralFilter(img,d=5,sigmaColor=100,sigmaSpace=1)
        #sigmacolorが大きいと、色がより均一になる,sigmaspaceが大きいと、より遠くのピクセルまで巻き込む
        #dは処理するときに見るピクセルの直径
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype(np.int16)

        # フィルタの大きさに基づいて領域を定義する？
        # em = EnergyMinimization()
        # gauregion = em.em_vj(src,seed_list,seed_img)

        for i,y,x in zip(range(len(seed_list[:,0])),seed_list[:,0],seed_list[:,1]):
            
            if rors[y,x] == 0 : # 新たな種
                    # print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y

                    # main processing
                    ror = self.growing_vj(img,x,y) # 新たなRegionの取得                
                    rors[ror==255] = 255 # Region of Rocksの更新
                    # cv2.imshow('growing',rors)
                    # cv2.waitKey(1)
                    # 上限だったら処理中止
                    if i == self.region_number:
                        break
            else: # すでに描画されたror
                continue
        
        return rors, img

    def growing_vj(self,src,x,y):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像

            param:      -> 
        '''

        img = copy.deepcopy(src)

        # 準備
        region_map = np.zeros([len(src[0]),len(src[:,0])]).astype(np.uint8)
        region_map[y,x] = 255
        seed = []
        seed.append([x,y])
        sx = copy.deepcopy(x)
        sy = copy.deepcopy(y)
        count = 1

        # region growing
        for i in xrange(100000):

            # 一つの領域の最大面積、分割が終わったら終了
            # if i == 500:
                # print 'size over'
                # break
            
            if i == len(seed):
                break
            
            # renew seed point
            x,y = seed[i]
            distance = math.sqrt( (y-sy)**2 + (x-sx)**2 )
            
            # 1pixelごとの処理
            for u,v in zip([x,x-1,x,x+1,x],[y-1,y,y,y,y+1]):

                # 処理中断条件                        
                if u == 0 or u >= len(src[0])-1 or v == 0 or v >= len(src[:,0])-1: # 画像の端だったらbreak
                    break

                # 領域拡張条件
                # elif 10 > abs(img[v,u] - img[y,x]) + distance or img[v,u]<20 and abs(img[v,u] - img[sy,sx])<50:
                # elif distance == 1.0 or distance == 0 and 25 > abs(img[v,u] - img[y,x]):
                elif 7 > abs(img[v,u] - img[sy,sx])  or img[v,u] < 40:

                    # renew region map
                    if region_map[v,u] == 0: # 新しい種だった場合
                        region_map[v,u] = 255
                        seed.append([u,v])
                        count += 1
                else:
                    continue

        return region_map

class EnergyMinimization():
    '''
    エネルギー最小化問題に基づいてregionを変更する
    '''

    def em_vj(self,src,seed_list,seed_img):
        
        dd = DisplayData()
        # フィルタの大きさは7とする
        # もっとも反応するフィルタは、岩よりも大きいはず
        # 11でやってる
        img = copy.deepcopy(src)

        img[321,236,2]=255
        img[321,236,1]=0
        img[321,236,0]=0

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[311:329,226:244]

        # ガウシアン関数の用意
        kernlen = 21
        nsig = 3
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = cv2.normalize(kernel, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)

        # 平均と分散の用意、kernelの大きさが11だから、それに基づいて確実に岩っぽいとことる
        ave1 = 250
        ave2 = 13



        # エネルギー最小化
        for i in range(18):
            for j in range(18):
                if abs(img[i,j]-250) > 30:
                    # print 'over'
                    img[i,j] = 0
                # else:
                    # print 'ok'
                print img[i,j]

        



        # dd.display_3D(kernel)
        cv2.imshow('kernel',kernel)
        cv2.imshow('src',src)
        cv2.imshow('img',img)

    def energy_minimization(self,src,region,sm):
        '''
        エネルギー最小化問題に基づいて入力されたregionを最適化する
        '''

        img = copy.deepcopy(src)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # regionをsrcにmaskする
        img[region==0] = 0
        cv2.imshow('imgs',img)

        ave = img.sum()/np.count_nonzero(img) # ok


        # もしあるピクセルが平均値から大きくはずれてたら0にする
        for i in range(400):
            for j in range(400):
                if img[i,j] - ave >20:
                    img[i,j] = 0




        #         if img[i,j]!=0:
        #             sumimg = sumimg + img[i,j]
        #             count = count + 1

        sample = [[1,2,3],[4,5,6],[7,8,0]]

        cv2.imshow('img',img)
        cv2.imshow('region',region)

class SaliencyMap():

    def __init__(self):
       
        # いくつのDoGを足し合わせるか,1-6まで
        self.scale = 1

    def SaliencyMap(self,img):
        '''
            入力された画像のSaliency Mapを求める
            args : img  -> 入力画像,3ch
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''
        # 前処理
        # img = self.PostProcessing(img)
        # cv2.imshow('post',img)

        # Saliency Mapの入手
        IntensityCM = self.GetIntensityCM(img)

        return IntensityCM

    def PostProcessing(self, img, number):
        '''
            前処理、フィルタ処理とか
            args : img  -> 入力画像
            dst  : SM   -> saliencymap,uint8のMat型1ch画像
            param: CM   -> 各特徴毎の正規化されたCouspicuity Map(顕著性map)
        '''
        for i in range(number):
            out = cv2.bilateralFilter(img,9,50,50)

        return out

    def GetIntensityCM(self,img):
        '''
           入力画像のIntensity Couspicuity Mapを求める
           argv: img         -> 入力画像
           dst : IntensityCM -> 強度の顕著性マップ
        '''
        # Get intensity image
        intensity_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # 前処理
        # intensity_img = self.PostProcessing(intensity_img, 5)

        # Create Gaussian pyramid　[0]元画像、全部で10枚
        intensity_pyr = self.GetGaussianPyramid(intensity_img) 
        intensity_pyr = np.array(intensity_pyr).astype(np.float)

        dog = cv2.absdiff(intensity_pyr[2],intensity_pyr[9])

        dog = cv2.normalize(dog, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
        # print dog.shape
        # dog = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)

        # Get 6 Difference of Gaussians, float64へ変換
        # intensity_dog = self.GetDoG(intensity_pyr) 
        # intensity_dog = np.array(intensity_dog).astype(np.float)

        # Get saliency map
        # intensity_sm = self.GetCM(intensity_dog)
        # sm = (intensity_sm/np.amax(intensity_sm) * 255).astype(np.uint8)

        return dog

    def GetIntensityImg(self,img):
        '''
            入力画像のIntensityImgを返す
            argv: img ->  3chのColorImg
            dst : IntensityImg -> 1chのImg
        '''
        b,g,r = cv2.split(img)
        IntensityImg = b/3. + g/3. + r/3.
        print 'rgb',np.amax(IntensityImg)
        R,G,B,Y = self.GetRGBY(img) # float
        # IntensityImg = r
        print 'r',np.amax(IntensityImg)

        return IntensityImg

    def GetGaussianPyramid(self,img):
        '''
            入力された画像のgaussian pyramidを計算する
            args : img    -> Mat,Uint8,1chの画像
            dst  : pyr  -> 元画像合わせて10階層のgaussian画像が入ったlist，それぞれargsと同じサイズ
                           pyr[i].shapeは，src.shape / 2**i
        '''
        # 準備
        pyr  = range(10)
        pyr[0]  = img 

        # Create Gaussian pyramid
        for i in range(1,10):
            pyr[i] = cv2.pyrDown(pyr[i-1])
        
        # Resize pyramid 
        for i in range(10):
            pyr[i] = cv2.resize(pyr[i],(len(img[0]),len(img[:,0])), interpolation = cv2.INTER_LINEAR)
        
        return pyr

    def GetDoG(self,pyr):
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
      
    def GetCM(self,dog):
        '''
            DoGからSMを形成する
        '''

        sm = np.zeros_like(dog[0])

        for i in range(self.scale):
            sm = sm + dog[i]
            # cv2.imshow('dog%s'%i,dog[i]/np.amax(dog[i]))

        return sm

class DisplayData():

    def display_histgram(self,img):
        '''
            処理の概要
            args :      -> 1ch or 3ch
            dst  :      -> 
            param:      -> 
        '''
        hist=img.ravel()
        plt.hist(hist,256,[0,256])
        plt.xlim([0,256])
        plt.pause(0.01)

    def display_3D(self,img):
        '''
            入力画像を3Dで表示する
            args: 1ch image
        '''
        # データの準備
        x = np.arange(0, len(img[0]), 1)
        y = np.arange(0, len(img[1]), 1)
        X, Y = np.meshgrid(x, y) 
        Z = img

        # plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(X,Y,Z)

        # 設定
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')        
        ax.set_zlabel('intensity')
        ax.set_zlim(0, 300)
        ax.set_title('Image')
        ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap = 'jet',linewidth=0)
        # ax.plot_wireframe(X,Y,Z, cmap = 'Greys', rstride=10, cstride=10)

        plt.pause(.001) # これだけでok
        # plt.show()


################################################################################
# main
################################################################################
if __name__ == '__main__':

    # read class
    rd = RockDetection()

    # get image
    # img = cv2.resize(cv2.imread('../../image/rock/spiritsol118navcam.jpg'),(512,512))
    # img = cv2.resize(cv2.imread('../../image/rock/sol729.jpg'),(512,512))
    # img = cv2.resize(cv2.imread('../../image/rock/11.png'),(512,512))
    img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')
    img = img[400:800,400:800]
    # img = img[230:300,230:300]
    # main processing
    rd.rock_detection(img)
    cv2.waitKey(-1)
    '''メモ'''
        # seedを横に増やすとき
        # row = (np.zeros((len(seed_img3),1))).astype(np.uint8)
        # shift = np.append(seed_img3[:,1:512],row,1)
        # seed_img3[shift==255] = 255