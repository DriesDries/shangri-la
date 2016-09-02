 # -*- coding: utf-8 -*-
'''
    Test function
    
    Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

def draw():
    for seed in seed_list:
            size = 5 + scale_img[tuple(seed)] * 2
            print size, seed, seed[0]-size/2, seed[1]-size/2, seed[1]+size/2, seed[1]+size/2
            cv2.rectangle(img,(seed[1]-size/2,seed[0]-size/2), (seed[1]+size/2,seed[0]+size/2),(100,100,255),thickness=1)


def chi_square_distance():
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    
    a = [50, 50, 50, 50]
    b = [40, 40, 40, 40]

    dis = stats.chisquare(a, b) # bで正規化してる
    print dis[0] # カイ二乗距離
    print dis[1]

def get_filter_bank(img):
    '''
        MR8 filterbankの取得
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    # 角度の準備
    
    sigmas_x = [1,2,4]
    sigmas_y = [3,6,12]

    rad = np.arange(0,151,30).astype(float) - 90
    angles = np.pi * rad / 180

    bank = []

    kernels = []
    for angle in angles:
        for sigmax, sigmay in zip(sigmas_x, sigmas_y):
            for frequency in (0.05, 0.25):
                kernel = np.real(filters.gabor_kernel(frequency, theta=angle,
                                              sigma_x=sigmax, sigma_y=sigmay))
                kernels.append(kernel)
                print angle,sigmax,sigmay,frequency, kernel.shape

    # gau_kernel = 
    # log_kernel = 

    # kernels.append((gau_kernel, log_kernel))

    kernels = np.array(kernels)
    print kernels.shape


    return kernels

def get_binary_ror(label_img):
    ## Get binary image of Region of Rocks
    size = label_img.shape[0]
    true_ror = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            # Soil
            if label_img[i,j,0] == label_img[i,j,1] == label_img[i,j,2]:
                pass
            # Rock
            else:
                true_ror[i,j] = 255

    return true_ror


class RegionGrowing():
    
    def __init__(self):
        
        # パラメータ
        self.thresh_rg = 1     # 結合条件の閾値
        self.thresh_sm = 20
        self.region_number = 20000
        self.max_size = 1000

    def main(self, img, seed_list, seed_img, scale, scale_number, responses):
        '''
            Region Growingのメイン
            args : src          -> img,3ch
                   x,y          -> seed pointの座標
                   scale        -> 画像形式
            dst  : region_map   -> 領域が255,他が0の1ch画像
                   masked_img   -> srcにmaskを重ねた画像
            param:      -> 
        '''        

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint16)
        gauimgs = self.get_gau_image(scale_number) # 0-1に正規化,float64 # 複数スケールのgaussian imageの用意
        rors = np.zeros_like(seed_img).astype(np.uint8)

        for i, y, x in zip( range(len(seed_list[:,0])) , seed_list[:,0], seed_list[:,1]):
            print '%s'%(i+1),'/%s'%len(seed_list[:,0]),'個目 x=',x,'y=',y            
            
            if rors[y,x] == 0: # 新たな種
                ror = self.growing_vj(img, x, y, gauimgs[scale[y,x]], responses) 
                rors[ror==255] = 255 # renew rors
                cv2.imshow('rors',rors)
                cv2.waitKey(1)
         
        return rors

    def growing_vj(self, src, ori_x, ori_y, gauimg, responses):
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
            
            if i >= 1000:
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
                # print stats.chisquare(responses[:,v,u], responses[:,y,x])[0]
                if abs(stats.chisquare(responses[:,v,u], responses[:,y,x])[0]) < 0.1:

                # if 5 > E1:
                    # print stats.chisquare(responses[:,v,u], responses[:,y,x])[0]

                # if (10 > E1) and dis[texton_map[u,v],texton_map[x,y]]<0.1:
                # if texton_map[v,u]==texton_map[y,x-1] :
                # if (10 > E1) and texton_map[v,u]==texton_map[y,x-1] :
                
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

    def get_gau_image(self,scale):
        '''
            scaleの配列に基づいた複数のガウシアン関数を用意する
        '''

        kernels = []

        sigma = range(12,2,-1) # 分散の定義

        for i in range(scale):
            size = 40 # kernelのサイズ
            nsig = sigma[i]  # 分散sigma
            interval = (2*nsig+1.)/(size)
            x = np.linspace(-nsig-interval/2., nsig+interval/2., size+1) # メッシュの定義

            kern1d = np.diff(st.norm.cdf(x)) # 多分ここでガウシアンにしてる,1次元
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d)) # 二次元にしてる
            kernel_raw = cv2.normalize(kernel_raw, 0, 1, norm_type = cv2.NORM_MINMAX)
            kernel_raw = abs(1-kernel_raw)
            kernels.append(kernel_raw)

        return kernels



if __name__ == '__main__':

    # chi_square_distance()

    img = cv2.imread('../../data/g-t_data/original/spirit050-2.jpg')
    img = cv2.imread('../../../Dropbox/ファイル 2016-08-30 16 26 08.png')
    img = get_binary_ror(img)
    # cv2.imwrite('../fasda.png',img,[int(cv2.cv.CV_IMWRITE_PNG_COMPRESSION), 0])
    # img = img[400:800,400:800]

    a = np.array([1,1,1,1])
    b = np.array([3,3,3,3])

    l = np.linalg.norm(b-a)
    print l

    # chi_square_distance()


