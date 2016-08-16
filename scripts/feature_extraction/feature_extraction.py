# -*- coding: utf-8 -*-
'''
    入力画像の特徴抽出を行って特徴を返す

    src: img -> 元画像
         ror -> 岩領域が非0となった画像
    dst: features -> 各行にそれぞれの特徴が入ってる
            - features[:,0] : compositions   -> 各岩領域の色の平均
            - features[:,1] : shape          -> fittingした楕円の短径と長径の比
            - features[:,2] : size           -> 楕円の長径+短径

            *** 以下は使用していない特徴 ***
            - p_sizes : rorの領域ごとのpixelの数

    Usage: $ python template.py <argv>

    ・組成を抽出する前に小さい領域は削除する
        - さらにerodeとかする
    ・sizeを楕円の面積にする
    ・極端に大きかったりするのは？
        - 大小でRD変えるなら削除するのもありかも
''' 

import cv2, os, sys, copy, time
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk
    
def main(src, ror_ori):
    '''
        画像を入力すると特徴を返す
        args : src  -> 元画像
               ror_ori  -> 岩領域が!=0となった画像
        dst  : size, shape, composition -> 
        param:      -> 
    '''
    ror = copy.deepcopy(ror_ori)
    

    compositions, ror_new = get_composition(src,ror_ori)
    print len(compositions)

    # 楕円フィッティングを行い、楕円からFEを行う   
    ellipses = fit_ellipse(src,ror_new)
    # centers, sizes, shapes = get_feature(ellipses)
    # print len(ellipses), len(sizes)

    # modified_sizes = modify_sizes(centers,sizes,src.shape[0])
    
    # plt.hist(compositions, bins=100)
    
'''
    sizes = sizes.reshape((sizes.shape[0],1))
    shapes = shapes.reshape((shapes.shape[0],1))
    
    # 特徴の結合
    features = np.hstack((shapes, sizes, shapes))

    print 'ellipse number = ',len(ellipses)
    
    return features
'''

def get_composition(src,ror):
    '''
        各領域の画素値の平均を求め、配列にして返す。
    '''        

    # 各岩領域にlabelをつける
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    labels, num = sk.label(ror, return_num = True)
    print num
    compositions = []
    p_sizes = []

    for i in range(1,num,1): # 非岩領域以外で

        if np.sum(labels==i) >30: # 領域の閾値
            compositions.append(np.sum(img[labels==i]) / np.sum(labels==i)) # (各領域の輝度値の合計) / (それぞれの領域のサイズ)
            p_sizes.append(np.sum(labels==i))
        
        else: # 領域が小さいとき
            labels[labels==i]=0


    return compositions, labels

def modify_sizes(centers, sizes, img_size):
    '''
        sizeをcenterに合わせて修正して返す
    '''

    dis = get_distance_matrix(pan=70, fov=40, height=1.5, rows=img_size)
    modified_sizes = []

    for i,c in enumerate(centers[:,0]):
        modified_sizes.append(sizes[i] * dis[i])

    return modified_sizes

def fit_ellipse(img, labels):
    '''
        rorから輪郭を抽出して返す
        args : img -> 元画像
               contours -> 輪郭が入った配列.contours[i]が一つの輪郭を表す
        dst  : ellipses -> 楕円fittingしたパラメータ。ellipses[i]がcontours[i]の輪郭にfitした楕円
        param:      -> 
        (400, 400) uint8
    '''
    
    # 輪郭の抽出, contours[i]にひとつづつ輪郭の座標が入る
    labels = labels.astype(np.uint8)
    labels[labels!=192]=0
    # cv2.imshow('labels',labels)
    # cv2.waitKey(-1)
    ellipses = []
    con = np.zeros_like(labels).astype(np.uint8)
    ell = np.zeros_like(labels).astype(np.uint8)

    for i,label in enumerate(range(1,np.amax(labels))):
        # print 'i',i,np.sum(labels==label)

        if np.sum(labels==label)>0: # もし領域があったら
            print 'draw!!!'
            con = np.zeros_like(labels).astype(np.uint8)

            con[labels==label] = 255
            con[con!=255] = 0
            '''一つの岩領域から複数の楕円が検出されてしまう'''
            contour, hierarchy = cv2.findContours(con, mode=1, method=2)
            # print contour
            cv2.drawContours(ell,contour,-1,(255,255,255),2)
            print 'ellipse',i, len(contour)
            print contour
            ellipse = cv2.fitEllipse(contour)

            # cv2.ellipse(ell,ellipse,(0,0,255),1)
            cv2.imshow('ell',ell)
            cv2.imshow('con',con)
            cv2.waitKey(-1)

        #     ellipses.append(ellipse)
            
            ell[ell != 0] = 0
            con[con != 0] = 0



        # else: # 小さすぎると楕円フィッティングできない
        #     continue

    print 'contours',i
    return ellipses

def get_feature(ellipses):
    '''
        入力された複数の領域のそれぞれの形状と大きさの抽出を行う
        args :      -> 
        dst  :      -> 中心座標、大きさ、形状
        param:      -> 
    '''
    centers  = []
    sizes  = []
    shapes = []

    for i, ellipse in enumerate(ellipses):
        
        centers.append((ellipse[0]))             # 楕円の中心座標
        shapes.append(ellipse[1][0]/ellipse[1][1])  # 長径と短径の比
        sizes.append(ellipse[1][0] + ellipse[1][1]) # 長径と短径の和
    
    centers = np.array(centers).astype(np.uint16) # float -> int

    return np.array(centers), np.array(sizes), np.array(shapes)

def modify_parameter(centers, sizes, shapes):
    pass

def get_distance_matrix(pan, fov, height, rows):
    '''
        画像と同じ行数で一列の配列を返す。
        それぞれの行にはcameraからcenterまでの距離が入ってる。

        src: pan       -> カメラの傾き   [radian]
             fov       -> カメラの視野角 [radian]
             height    -> 地面からカメラまでの高さ [m]
             rows      -> 画像の縦方向のピクセル数 [pixel]

        dst: dis       -> 距離の真値
             dis_ratio -> dis[row] / dis[len(rows)]の値(画像中心からの距離の比)
    '''
    
    # Convert radian -> degree
    pan = 1. * pan * np.pi /180
    fov = 1. * fov * np.pi /180
    dis = []
    dis_ratio = []

    for row in range(rows-1, -1, -1):
        # dis.append( height * np.tan(1.*row*fov/rows + pan - 1.*fov/2))
        dis_ratio.append( np.tan(1.*row*fov/rows + pan - 1.*fov/2) / np.tan(pan))

    return dis_ratio


if __name__ == '__main__':

    # img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')
    # img = img[400:800,400:800]        
    
    # sample image
    img = cv2.imread('../../../image/rock/spiritsol118navcam.jpg')
    img = img[400:800,400:800]
    l_ror = cv2.imread('../../../image/sample/sample1.png',0)
    s_ror = cv2.imread('../../../image/sample/sample2.png',0)

    main(img, s_ror)

    # fig = plt.figure(figsize=(16,9))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    
    # ax1.hist(sizes, bins=100)
    # ax2.hist(modified_sizes, bins=100)
    plt.show()
    # sm = cv2.dilate(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size))
    # sm = cv2.erode(sm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size))
  