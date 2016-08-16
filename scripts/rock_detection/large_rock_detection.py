# -*- coding: utf-8 -*-
'''
Rock Detection based on Saliency Map and Region Growing Algolithm
Usage: $ python rock_detection.py
argv : Image
dst  : Region of Rocks, 1ch
''' 

import time, math, copy, os, sys

import cv2
import numpy as np
import skimage.measure as sk
import skimage.segmentation as skseg
import scipy.stats as st

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def main(img):
    ror = rd.large_rd(img)
    return ror

class RockDetection():
    
    def __init__(self):
        
        self.min_size = 10        # 最小領域の大きさ  
        
        #　paramater for sm
        self.thresh_sm_maxima = 150   # 極大値求めるときのsmの閾値
        self.kernel_size = (11,11)  # dilateとerodeのkernelの大きさ
        
        # paramater for vj
        self.sun_direction = np.pi
        self.vj_thresh_value = 180

    def large_rd(self,src):
        '''
            ---algolithm---
            まず、元画像からSaliencyMapを生成し、生成したSMから閾値以上の全極大値を求め、その極大値をseedとして保存する。そして、そのseedから領域拡張法に基づいて領域分割を行うことで、岩領域を算出する。listはy,x(画像中で)の多行二列
        '''
        img = copy.deepcopy(src)

        # Seed Acquisition
        sm = smap.SaliencyMap(img)
        seed_img, seed_list = self.get_seed(sm,self.thresh_sm_maxima)
        # seed_img, seed_list = self.detect_maxima(sm,self.thresh_sm_maxima) # sano-algo
        print 'large seed number = ',len(seed_list)

        # Region Growing
        # smの正規化, 値が小さいとhigh saliency
        sm_inv = (255 - sm).astype(np.float32)
        sm_inv = cv2.normalize(sm_inv, 0, 1, norm_type = cv2.NORM_MINMAX)

        # Region Growing　Algorithm
        ror, bil_img = rgrow.rg_sm(img, sm, seed_list)

        ###結果の表示###########################################################
        # cv2.imshow('smseed',seed_img)
        # cv2.imshow('sm',sm)

        # sm[sm<self.thresh_sm_maxima]=0
        cv2.imshow('Original Saliency Map',self.display_result(sm,seed_img,'fill','g'))
        cv2.imshow('sm_ror',self.display_result(sm,ror,'fill','r'))

        # cv2.imshow('sm_ror_bil',self.display_result(bil_img,sm_ror,'fill','r'))
        # cv2.imshow('sm_seed_img',self.display_result((o*0.8).astype(np.uint8),sm_seed_img,'fill','r'))
        # cv2.imshow('seed',seed_img)
        result = self.display_result((src*0.9).astype(np.uint8),ror,'fill','r')
        cv2.imshow('ror',self.display_result(result,seed_img,'fill','g'))
        #######################################################################

        return ror

    def get_seed(self,ori_sm,thresh):
        '''
        入力された画像から種を求める
        '''

        # 閾値処理と領域分割
        sm = copy.deepcopy(ori_sm)
        sm[sm < thresh] = 0
        sm[sm != 0] = 255
        labels, num = sk.label(sm, return_num = True) 

        seed_list = []
        seed_img = np.zeros_like(sm)
        mask = np.zeros_like(sm).astype(np.uint8)
        
        # 各領域の最大値を求める
        for i in range(1,num+1):
           
            sm = copy.deepcopy(ori_sm) # 初期に戻す
            sm[labels!=i] = 0 # これで残った領域の最大値求める
            
            y = np.argmax(sm)/len(sm)
            x = np.argmax(sm)%len(sm)

            if sm[y,x]!=0: # 中に空いた穴はだめ
                seed_img[y,x] = sm[y,x]
                seed_list.append([y,x])
            
            # cv2.imshow('sm',sm)
            # cv2.waitKey(0)

        seed_list = np.array(seed_list)
        
        return seed_img, seed_list

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
        
        # 岩領域の画像用意
        rors = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)).astype(np.uint8)

        # 平滑化
        # for i in range(5):
            # img = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img16 = img.astype(np.int16)
        sm16 = sm.astype(np.int16) 
        for i,y,x in zip(range(len(seed_list[:,0])),seed_list[:,0],seed_list[:,1]):

            if rors[y,x] == 0: # 新たな種
                # print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y
                
                rors[y,x] == 255
                ror = self.growing_sm(img16,sm16,x,y) # 新たなRegionの取得
                rors[ror==255] = 255 # Region of Rocksの更新
                
                # cv2.imshow('growing',rors)
                # cv2.waitKey(0)

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
        ave = img[y,x]
        region = 1

        sy = copy.deepcopy(y)
        sx = copy.deepcopy(x)
        var2 = 0

        # ave = (src[sy-1,sx-1]*1. + src[sy,sx-1] + src[sy+1,sx-1])/3

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

            for u,v in zip([x,x-1,x+1,x],[y-1,y,y,y+1]):
                # 領域拡張の条件式
                # E = sm[v,u] + abs(img[v,u]-ave) + abs(img[v,u]-img[y,x])
                # E = abs(img[v,u] - img[y,x]) + 35*sm[v,u]
                E1 = abs(img[v,u] - img[y,x])
                E2 = (sm[y,x] - sm[v,u])
                E3 = sm[v,u]
                E4 = abs(img[v,u] - ave)
                if E2 >= 0:                
                # if E4 < 60 and E2 > 0:                

                    if region_map[v,u] == 0: # 拡張
                        region_map[v,u] = 255
                        seed.append([u,v])
                        count += 1

                    # それまでの岩領域の平均値の計算
                    region = region + 1
                    sum_val = sum_val + img[v,u]
                    ave = sum_val / region
                    # print ave

                else:
                    continue
        return region_map

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

class etc(): 
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

'''Read Class'''
rd = RockDetection()
smap = SaliencyMap()
rgrow = RegionGrowing()
dd = DisplayData()


if __name__ == '__main__':

    ## Image acquisition
    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')

    # main processing
    main(img)

    cv2.waitKey(-1)
    