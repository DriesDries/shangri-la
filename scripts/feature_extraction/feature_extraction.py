# -*- coding: utf-8 -*-
'''
  
    Return features of input binary RoR-image

    Usage: $ python feature_extraction.py
    
    src: img -> 元画像
         ror -> 岩領域が非0となった画像
    dst: features -> 各行にそれぞれの特徴が入ってる
            - features[:,0] : compositions   -> 各岩領域の色の平均
            - features[:,1] : psize          -> 領域のピクセル数
            - features[:,2] : size           -> 楕円の長径+短径
            - features[:,3] : shape          -> fittingした楕円の短径と長径の比
            - features[:,4] : center_x       -> 楕円の中心座標x
            - features[:,5] : center_y       -> 楕円の中心座標y


    ToDO:
        ・sizeを楕円の面積にする
        ・小さい領域には楕円をfittingすることができない
            - そもそもする必要はあるのか？
''' 
import time
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk
    
def main(img, ror):
    '''
        画像を入力すると特徴を返す
        args : src  -> 元画像
               ror_ori  -> 岩領域が!=0となった画像
        dst  : size, shape, composition -> 
        param:      -> 
    '''
    
    ## labeling ## 小さい領域を削除
    labels = get_labels(ror)

    ## Get feature
    compositions = get_composition(img, labels)

    # Get geometric features by ellipse fitting   
    centers, sizes, shapes, psizes = geometric_features(img, labels, modify='OFF')

    ## merge features
    features = np.hstack((compositions, psizes, sizes, shapes, centers))

    return features

def get_labels(ror, minsize=20):
    '''
        ・各領域にラベルをつける
        ・0の領域にはラベルをつけない
        ・minsizeより小さい領域は削除する
    '''
    labels = sk.label(ror, return_num = False, background=0)

    for i in range(1, np.max(labels)+1):
        if np.count_nonzero(labels[labels==i]) < minsize:
            ror[labels == i] = 0 
    
    ## Generate label images from only large regions
    large_labels = sk.label(ror, return_num = False, background=0)
    # print np.nonzero(large_labels==0)
    return large_labels

def get_composition(img, labels):
    '''
        各領域の画素値の平均を求め、配列にして返す。
    '''        
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compositions = []

    for i in range(0, np.max(labels)+1): # 非岩領域以外で
        compositions.append(np.sum(img[labels==i]) / np.count_nonzero(img[labels==i])) # (各領域の輝度値の合計) / (それぞれの領域のサイズ)
        
    compositions = np.array(compositions)
    compositions = compositions.reshape((compositions.shape[0],1))

    return compositions

def geometric_features(img, labels, modify):
    '''
        入力された複数の領域のそれぞれの形状と大きさの抽出を行う
        args :      -> 
        dst  :      -> 中心座標、大きさ、形状
        param:      -> 
    '''
    centers  = []
    sizes  = []
    shapes = []
    psizes = []

    ## Ellipse fitting
    ellipses = fit_ellipse(img, labels)

    ## Get geometry features
    for i, ellipse in enumerate(ellipses):

        centers.append((ellipse[0]))             # 楕円の中心座標
        shapes.append(ellipse[1][0]/ellipse[1][1])  # 長径と短径の比
        sizes.append(ellipse[1][0] + ellipse[1][1]) # 長径と短径の和
        psizes.append(np.count_nonzero([labels==i]))
    
    ## modify region sizes based on distance from camera
    centers = np.array(centers).astype(np.uint16) # float -> int
    if modify == 'ON':
        sizes = modify_sizes(centers, sizes, img.shape[0])

    shapes, sizes, psizes = map(lambda x: np.array(x).reshape( np.array(x).shape[0],1), [shapes, sizes, psizes])

    return centers, sizes, shapes, psizes

def fit_ellipse(img, labels):
    '''
        rorから輪郭を抽出して返す
        args : img -> 元画像
               contours -> 輪郭が入った配列.contours[i]が一つの輪郭を表す
        dst  : ellipses -> 楕円fittingしたパラメータ。ellipses[i]がcontours[i]の輪郭にfitした楕円
        param:      -> 
        (400, 400) uint8
    '''
    

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eximg = np.vstack((np.zeros(gimg.shape[0]), gimg, np.zeros(gimg.shape[0])))
    a = np.reshape(np.zeros(eximg.shape[0]), (np.zeros(eximg.shape[0]).shape[0],1))
    eximg = np.hstack((a, eximg, a))

    ellipses = []

    for i in range(0, np.amax(labels)+1):
        # print i, np.count_nonzero(labels==i)
        ## Initialization
        res = np.zeros_like(gimg).astype(np.uint8)
        res[labels==i] = 255

        res = np.vstack((np.zeros(res.shape[0]), res, np.zeros(res.shape[0])))
        a = np.reshape(np.zeros(res.shape[0]), (np.zeros(res.shape[0]).shape[0],1))
        res = np.hstack((a, res, a)).astype(np.uint8)

        contour, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contour) >= 2: # エラー処理
            print 'Detected {} ellipses from sigle rock'.format(len(contour))
        ellipse = cv2.fitEllipse(contour[0])
        ellipses.append(ellipse)

        ## draw contours
        # cv2.ellipse(img, ellipse, (0,0,255), 1)

    return ellipses

def modify_sizes(centers, sizes, img_size):
    '''
        sizeをcenterに合わせて修正して返す
    '''

    dis = get_distance_matrix(pan=70, fov=40, height=1.5, rows=img_size)
    modified_sizes = []

    for i,c in enumerate(centers[:,0]):
        modified_sizes.append(sizes[i] * dis[c])

    return np.array(modified_sizes)

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

    # sample image
    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')
    ror = cv2.imread('../../../data/g-t_data/label/spirit118-1.png',0)

    ## feature extraction
    main(img, ror)

    ## Draw result
    # fig = plt.figure(figsize=(16,9))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.hist(sizes, bins=100)
    # ax2.hist(modified_sizes, bins=100)
    # plt.show()