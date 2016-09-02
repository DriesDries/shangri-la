# -*- coding: utf-8 -*-
'''
    
    Usage: $ python main.py

    * 解析的？にどういう場合に岩である確率が高いかを調査し、以下のアルゴリズムでさらに精度を上げたい
    * Autonomous Scienceに適した課題と手法にしたい
         - Adaboost -> 弱識別器を組み合わせることで高精度な分類をする
         - Self-training -> 過去のデータで信頼性の高いものを教師データとして学習する半教師あり学習のアルゴリズム？

''' 
import time
import cv2
import numpy as np

import rock_detection.rock_detection as rd
import feature_extraction.feature_extraction as fe
import modeling.modeling as md
import anomaly_detection.anomaly_detection as ad

import evaluate.seed_eval as seedev
import evaluate.rg_eval as rgev

def main(img, true_ror):
    '''
        args : src     ->  3ch, input image
               true_ror -> 1ch, ground-truth binary image
    '''
    ## get params
    params = get_params('test')
    scores = []

    for i, param in enumerate(params):
        
        ## Target Detection and Environment
        ror, seed = rd.main(img, param) ; print 'RD'
        features = fe.main(img, ror)    ; print 'FE'
        md.main(features)               ; print 'MD'
        ad.main()                       ; print 'AD'

        ## Evaluate results
        # score = rgev.main(img, ror, seed, param)
        # scores.append(score)
        # ev.main(score)

def get_params(mode):
    '''
        parameterの配列を返す

        params  ->  parameterがgridで入ってる
           - params[:,0] : threshold of tm
           - params[:,1] : sigma
           - params[:,2] : bias
           - params[:,3] : direction
    '''
    if mode == 'test':
        params = [[167, 4, 0, np.pi*7/4, 11, 0.09],[167, 4, 0, np.pi*7/4, 11, 0.09]]

    else:
        ## 配列
        # thresh = range(153, 220, 1)
        # sigma = [4,5]
        # bias = np.arange(0.1,0.6,0.1)
        # direction = [np.pi * 0.9, np.pi, np.pi*1.1]
        # dif2 = np.arange(0,0.3,0.03)
        # dif = np.arange(3,15,2)

        ## 定数
        threshes = [167,170]
        sigmas = [4]
        biases = [0]
        directions = [np.pi]
        th1s = [11]
        th2s = [0.09]

        params = []
        ## 固定したいものを上層に
        for sigma in sigmas:
            for thresh in threshes:
                for bias in biases:
                    for direction in directions:
                        for th1 in th1s:
                            for th2 in th2s:
                                params.append((int(thresh), int(sigma), bias, direction, th1, th2))


    return params

if __name__ == '__main__':

    # filename = 'spirit006-1.png'
    filename = 'spirit118-1.png'

    img = cv2.imread('../../data/g-t_data/resized/{}'.format(filename))
    true_ror = cv2.imread('../../data/g-t_data/label/{}'.format(filename),0)
    print 'Target image : {}'.format(filename)

    # main processing
    main(img, true_ror)
