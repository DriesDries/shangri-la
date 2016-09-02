# -*- coding: utf-8 -*-
'''
    Usage: $ python template.py <argv>

    面積に関する評価を行う
''' 
 
import os, sys, time, copy

import cv2
import numpy as np
import skimage.measure as sk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import rock_detection.rock_detection as rd
import evaluate.rd_eval as ev
import feature_extraction.feature_extraction as fe
import modeling.modeling as md
import anomaly_detection.anomaly_detection as ad


# fig = plt.figure(figsize=(16,9))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# ax3 = fig.add_subplot(222)
# ax4 = fig.add_subplot(224)


# ax1.set_xlim(-2,4)
# ax1.set_ylim(0,1)
# ax1.set_ylabel('precision or recall')
# ax1.set_title('Target selection performance')
# ax1.set_xticks([0,2])
# ax1.set_xticklabels(['Precision','Recall'])

# ax2.set_xlim(0,1)
# ax2.set_ylim(0,1)
# ax2.set_xlabel('precision')
# ax2.set_ylabel('recall')
# ax2.set_title('Detection performance')


def main(img, true_ror):
    '''
        args : src     ->  input,3ch-image
               true_ror -> ground-truth image, 1ch
        dst  : model   ->  model
               target  ->  target
        param: params  ->  parameterがgridで入ってる
                   - params[:,0] : threshold of tm
                   - params[:,1] : sigma
                   - params[:,2] : bias
                   - params[:,3] : direction
    '''
    params, variety = get_params()

    label_img = sk.label(true_ror,return_num = False)

    ## predの入手
    for param in params:
        print 'dif=',param[4],'dif2=',param[5]
        pred_ror, seed_img, texton_map = rd.main(img, param=param)
    # texton_map = cv2.imread('../texton_map.png',0)

        eval_area(label_img, pred_ror)
    # eval_txm2(label_img, texton_map)
    # eval_txm(true_ror, texton_map)

    # ax4.imshow(true_ror, cmap='gray')
    # ax4.set_title('original image')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.pause(-1)

def eval_txm2(label_img, texton_map):

    sizes = range(0,1001,50)
    texton_map = cv2.cvtColor(texton_map,cv2.COLOR_BGR2GRAY) 
    print np.max(texton_map),np.min(texton_map)


    for size in sizes:

        # 領域を削除
        region = del_small(label_img.copy(), size=size)

        if len(np.nonzero(region)) != 0:
            region_map = texton_map.copy()

            # 残った領域のtextonを表示
            region_map[region == 0] = 0

            regionten = region_map.flatten()
            rockten = regionten[np.nonzero(regionten)]

            ax1.hist(rockten, normed=1, bins=16)

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            ax1.set_title('Size : {} ~ {}'.format(size,size+50))
            x2 = ax2.imshow(region_map)
            fig.colorbar(x2, cax=cax)

            plt.pause(2)

        # else:
            # print 'nai!!!'

def eval_txm(true_ror, texton_map):
    '''
        texton mapに関する評価を行う
    '''
    texton_map += 1 # 最小が1
    txten = texton_map.flatten()

    # 岩に属してるピクセルを改めてplot
    rock_map = texton_map.copy()
    rock_map[true_ror==0] = 0
    rockten = rock_map.flatten()

    soil_map = texton_map.copy()
    soil_map[true_ror!=0] = 0
    plt.imshow(soil_map)
    plt.pause(-1)
    soilten = soil_map.flatten()

    ax1.hist(soilten[np.nonzero(soilten)],normed=1)
    ax2.hist(rockten[np.nonzero(rockten)],normed=1)
    ax1.set_title('soil')
    ax2.set_title('rock')
    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    tm = ax3.imshow(texton_map, cmap = 'jet')
    ax3.set_title('texton_map')
    fig.colorbar(tm, cax=cax)
    
def eval_area(true_ror, pred_ror):
    '''
        面積に対する評価をする
        [true_size, pred_size, score]の配列を返す
        被ってる要素のpredの番号のサイズをとってきて比較すればよさそう
        もし二つあったら、二つの合計を取る

        それぞれのiごとで、
            T     : 岩の大きさの真値
            P     : pred_rorの大きさの新地
            TnorP : 岩領域で検出できていないピクセル数
            TandP : 検出できているピクセル数
            PnorT : 後検出のピクセル数

            Pを求める！

    '''
    detect = 0
    total_pre = 0
    total_recall = 0

    pred_ror,num = sk.label(pred_ror, return_num = True)

    for i in range(1,np.max(true_ror)+1):

        TandP = np.count_nonzero(pred_ror[true_ror == i]) # pred_ror[true_ror == i]でzeroじゃなかった数 / iの領域のサイズ
        
        if TandP !=0: # もし検出できていれば

            ## Get P
            non = np.nonzero(pred_ror[true_ror == i])
            p = np.unique(pred_ror[true_ror == i][non]) 
            P = 0
            for i2 in p:
                P += (pred_ror == i2).sum()

            ## Get others
            T = (true_ror == i).sum()
            TnorP = (true_ror == i).sum() - np.count_nonzero(pred_ror[true_ror == i])
            PnorT = P - TandP

            ## Get score
            pre = 1. * TandP / P
            recall = 1. * TandP / T


            total_pre += pre
            total_recall += recall
            detect += 1
            ## Draw
            # plt.scatter(pre,recall)

            # plt.scatter(P,T)
            # print T,P,TandP,TnorP,PnorT

    print 'pre    = ',total_pre/detect
    print 'recall = ',total_recall/detect
    print 'f      = ',2. * total_recall/detect * total_pre/detect / (total_recall/detect+total_pre/detect) 
    plt.scatter(total_pre/detect,total_recall/detect,color='r')
    plt.xlabel('precision')
    

def get_params():

    dif2 = np.arange(0,0.3,0.03)
    dif = np.arange(3,15,2)

    thresh = range(140,241,10)
    sigma = [4,5]
    bias = np.arange(0.1,0.6,0.1)
    direction = [np.pi * 0.9, np.pi, np.pi*1.1]

    # もし定数にするとき
    thresh = [164]
    sigma = [4]
    bias = [0]
    direction = [np.pi*7/4]
    dif = [11]
    dif2 = [0.09]

    params = []

    # 固定したいものを上層に
    for s in sigma:
        for th in thresh:
            for b in bias:
                for d in direction:
                    for di in dif:
                        for di2 in dif2:
                            params.append((int(th), int(s), b, d, di, di2))

    return np.array(params), len(thresh)

def get_true_features(img, label_img):
    '''
        label画像から岩領域とその特徴を抽出する
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rock = np.zeros_like(gimg)
    features = []
    
    ## Get label
    labels, num = sk.label(true_ror, return_num = True)

    for val in range(1,num):

        ## Process by region
        rock[labels==val]=255

        ## Feature extraction
        non = np.nonzero(rock) # make tuple for speeding up
        center = int(1. * non[0].sum()/len(non[0]))
        psize = len(non[0])
        color = int(1. * gimg[non].sum() / psize)
        features.append(list((0, val, center, psize, color)))

        ## Initialization
        rock = np.zeros_like(rock)

    return true_ror.astype(np.uint8), labels, features

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

def del_small(label_img, size):
    '''
        指定した大きさ以外の領域は削除する
    '''
    for i in range(1,np.max(label_img)+1):
        if size <= np.count_nonzero(label_img[label_img == i]) <= size+50:
            continue
        else:
            label_img[label_img == i] = 0
            

    return label_img

if __name__ == '__main__':

    # img = cv2.imread('../../data/g-t_data/resized/spirit118-1.png')
    # true_ror = cv2.imread('../../data/g-t_data/label/spirit118-1.png',0)

    img = cv2.imread('../../data/g-t_data/resized/spirit006-1.png')
    true_ror = cv2.imread('../../data/g-t_data/label/spirit006-1.png',0)


    # main processing
    main(img, true_ror)

    cv2.waitKey(-1)
    