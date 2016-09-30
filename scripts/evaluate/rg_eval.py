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

def main(pred_ror, true_ror, display=False):
    '''
        args : src     ->  1ch, predicted rock regions image
               true_ror -> 1ch, ground-truth binary image
    '''

    ## 全体の評価
    convex = 1. * np.count_nonzero(pred_ror) / np.count_nonzero(true_ror)

    ## それぞれの評価
    pre, recall = eval_area(pred_ror, true_ror)

    if display == True:
        plt.scatter(pre, recall, color='r')
        plt.xlabel('Correct / Predicted region')
        plt.ylabel('Detected / Rock region')
        plt.show()
    
    return convex, pre, recall

def eval_area(pred_ror, true_ror):
    '''
        面積に対する評価をし、[true_size, pred_size, score]の配列を返す

        それぞれのiごとで、
            T     : 岩の大きさの真値
            P     : pred_rorの大きさの新地
            TnorP : 岩領域で検出できていないピクセル数
            TandP : 検出できているピクセル数
            PnorT : 後検出のピクセル数
    '''
    detect = 0
    total_pre = 0
    total_recall = 0

    pred = sk.label(pred_ror, return_num = False, background=None)
    true = sk.label(true_ror, return_num = False, background=0)

    for i in range(0, np.max(true_ror)+1):

        TandP = np.count_nonzero(pred[true == i]) # pred_ror[true_ror == i]でzeroじゃなかった数 / iの領域のサイズ
        
        if TandP !=0: # もし検出できていれば

            ## Get P
            non = np.nonzero(pred[true == i]) 
            p = np.unique(pred[true == i][non]) ## 被っている領域のpredの番号
            P = 0 # Initialization
            for i2 in p:
                P += (pred == i2).sum()

            ## Get others
            T = (true == i).sum()
            TnorP = (true == i).sum() - np.count_nonzero(pred[true == i])
            PnorT = P - TandP

            ## Get score
            pre = 1. * TandP / P
            recall = 1. * TandP / T
            
            ## renew total score
            total_pre += pre
            total_recall += recall
            detect += 1
            
            ## Draw
            # plt.scatter(pre, recall, color = 'b')
            # print T,P,TandP,TnorP,PnorT

    pre_ave    = 1. * total_pre   / detect
    recall_ave = 1. * total_recall/ detect

    return pre_ave, recall_ave
    
if __name__ == '__main__':

    ## sample image
    pred_ror = cv2.imread('../../data/ror.png', 0)
    true_ror = cv2.imread('../../data/g-t_data/label/spirit118-1.png', 0)

    # main processing
    main(pred_ror, true_ror, display=True)

    print 'pre    = ',total_pre/detect
    print 'recall = ',total_recall/detect
    print 'f      = ',2. * total_recall/detect * total_pre/detect / (total_recall/detect+total_pre/detect) 

    cv2.waitKey(-1)