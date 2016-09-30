# -*- coding: utf-8 -*-
'''
    
    Usage: $ python main.py

    * 解析的？にどういう場合に岩である確率が高いかを調査し、以下のアルゴリズムでさらに精度を上げたい
    * Autonomous Scienceに適した課題と手法にしたい
         - Adaboost -> 弱識別器を組み合わせることで高精度な分類をする
         - Self-training -> 過去のデータで信頼性の高いものを教師データとして学習する半教師あり学習のアルゴリズム？

    * display_modelsがなんか正しくなさそう
    * i = 0とかで分けて結果に文字表示
    * sizeごと、距離ごとの評価

''' 
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import skimage.measure as sk
from mpl_toolkits.mplot3d import Axes3D

import rock_detection.rock_detection as rd
import feature_extraction.feature_extraction as fe
import modeling.modeling as md
import anomaly_detection.anomaly_detection as ad

import evaluate.seed_eval as seedev
import evaluate.rg_eval as rgev

# fig = plt.figure()
# ax = fig.add_subplot(111)


def main(img, true_ror):
    '''
        args : src     ->  3ch, input image
               true_ror -> 1ch, ground-truth binary image
    '''
    ## get params
    params = get_params('test')
    scores = []
    pre2s = []
    recall2s = []
    
    ## True feature
    true_features = fe.main(img, true_ror.copy())
    true_models   = md.main(true_features, N=10)
    composition, size, shape = ad.main(true_features, true_models)    ; print 'Complete AD'
    raw_input()


    # ad.main()                       ; print 'Complete trueAD'
    for i, param in enumerate(params):
      if i == 1:  
        
        '''Target Detection'''
        print 'Start detection'
        pred_ror, pred_seed = rd.main(img, param=param)
        pred_features       = fe.main(img, pred_ror)      
        pred_models         = md.main(pred_features, N=10)
        # composition, size, shape = ad.main(pred_features, pred_models)    ; print 'Complete AD'
        
        '''Evaluate'''
        # pre, recall, f, detect_img = seedev.main(pred_seed, true_ror)
        # convex, pre2, recall2      = rgev.main(pred_ror, true_ror)
        
        '''Display results'''
        # display_3d(pred_features)
        display_3d(img, pred_features)
        display_models(pred_models[0:4], true_models[0:4], pred_features, true_features)
        # display_result(true_ror, param, pred_seed, detect_img, pre, recall, f, i, len(params))
        # display_result2(img, detect_img, pred_ror, true_ror, pred_seed) # 画像のスコア
        # display_result3(convex, pre2, recall2, param)
        # pre2s.append(pre2)
        # recall2s.append(recall2)
    plt.show()
    # ax.plot(pre2s, recall2s, "-o", color = 'b' )
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)

def display_3d(img,features,target):
    '''
        入力画像を3Dで表示する
        args: X,Y,Z
        dst : None
    '''
    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.set_xticks([])
    ax.set_zticks([])
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')        
    ax.set_zlabel('z')
    ax.set_zlim(-0.5,0.5)
    ax.set_title('Map')

    ## Camera paremeters    
    H = 1.4
    FOV = 20.
    P = 400.
    pan = 60.

    '''imgの描画'''
    X = range(0,400,2)
    Y = range(400,0,-2)
    X,Y = np.meshgrid(X,Y)
    Y2 = H  * np.tan( ( 1.*Y*FOV/P + pan - FOV/2 ) *np.pi/180)
    X2 = Y2 * ( np.tan(abs(  FOV * X/P    ) *np.pi/180) - np.tan((FOV/2) *np.pi/180) )

    if len(img.shape) >=3:
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gimg = img.copy()
    gimg = cv2.resize(gimg,(len(X),len(Y)))

    # ax.scatter(X2, Y2, -0.2, s=20, c=gimg, cmap='gray', linewidths=0 )

    '''featuresの描画'''
    # X = abs(400 - features[:,4])
    X = features[:,4]
    Y = abs(400 - features[:,5])
    Y2 = H  * np.tan( ( 1.*Y*FOV/P + pan - FOV/2 ) *np.pi/180)
    X2 = Y2 * ( np.tan(abs(  FOV * X/P  ) *np.pi/180) - np.tan((FOV/2) *np.pi/180) )
    sizes = features[:,1] * np.sqrt(X2**2 + Y2**2 + H ** 2)

    # print X
    print X2
    print 'target',target
    print 'max',features[np.argmax(features[:,0]),2],np.max(features[:,2])
    print 'x2',X2[target], Y2[target]
    ax.scatter(X2, Y2, sizes/np.max(sizes)/10, s=sizes/4, marker='o', c=features[:,0], cmap='gray', linewidths=1 )
    ax.scatter(X2[target], Y2[target], sizes[target]/np.max(sizes)/10, s=sizes/4, marker='o', c='r', linewidths=1 )


    plt.show()

def display_models(pred_models, true_models, pred_features, true_features):
    '''
            - features[:,0] : compositions   -> 各岩領域の色の平均
            - features[:,1] : psize          -> 領域のピクセル数
            - features[:,2] : size           -> 楕円の長径+短径
            - features[:,3] : shape          -> fittingした楕円の短径と長径の比
            # - features[:,4] : center_x       -> 楕円の中心座標x
            # - features[:,5] : center_y       -> 楕円の中心座標y
    '''

    fig = plt.figure(figsize = (16,9))  
    N = len(pred_models[0].weights_)

    x_start = []
    x_stop = []

    for i in range(len(pred_models)):
        x_start.append(min( min(pred_features[:,i]), min(true_features[:,i]) ))
        x_stop.append(max( max(pred_features[:,i]), max(true_features[:,i]) ))
        # y_start =
        # y_stop  = 


    for i, EM in enumerate(pred_models):
        ax = fig.add_subplot(len(pred_models), 2, 2 * i + 1)
        x = np.linspace(start=x_start[i], stop=x_stop[i], num=1000)
        y = 0
        ps = range(N)
        p = 0
        
        for k in range(N): # それぞれのガウス分布を描画

            ps[k] = EM.weights_[k] * mlab.normpdf(x, EM.means_[k,0], math.sqrt(EM.covars_[k][0][0]))
            p += ps[k]
            plt.plot(x, ps[k], color='orange')

        if EM.converged_ == True: # 収束してたら描画
            plt.plot(x, p, color='red', linewidth=3)
            plt.hist(pred_features[:,i], bins = 30, color='dodgerblue', normed=True)
        
        else: # 収束しなかった場合
            print '!!!Cannot converge!!!'
            # score = EM.score(pred_features).sum()

    num = 100 * len(true_models) + 21
    for i, EM in enumerate(true_models):
        ax = fig.add_subplot(len(true_models), 2, 2 * i + 2)
        x = np.linspace(start=x_start[i], stop=x_stop[i], num=1000)
        y = 0
        ps = range(N)
        p = 0
        
        for k in range(N): # それぞれのガウス分布を描画

            ps[k] = EM.weights_[k] * mlab.normpdf(x, EM.means_[k,0], math.sqrt(EM.covars_[k][0][0]))
            p += ps[k]
            plt.plot(x, ps[k], color='orange')

        if EM.converged_ == True: # 収束してたら描画
            plt.plot(x,p,color='red',linewidth=3)
            plt.hist(true_features[:,i], bins = 30, color='dodgerblue', normed=True)
        
        else: # 収束しなかった場合
            print '!!!Cannot converge!!!'

def display_result(true_ror, param, seed_img, detect_img, pre, recall, f, i, atp_num):

    rock = np.max(sk.label(true_ror, return_num = False, background=0))+1
    correct_seed_img = seed_img.copy()
    correct_seed_img[true_ror==0] = 0
    detect_img[detect_img != 255] = 0
    detect = np.max(sk.label(detect_img, return_num = False, background=0))+1

    print '======================='
    print 'Seed threshold : {}'.format(int(param[0]))
    print 'Gabor sigma    : {}'.format(param[1])
    print 'Gabor bias     : {}'.format(param[2])
    print 'Sun direction  : {}π'.format(param[3]/np.pi)
    print 'Threshold1     : {}'.format(param[4])
    print 'Threshold2     : {}'.format(param[5])
    print 'Rock quantity  : {}'.format(rock)
    print 'Seed quantity  : {}'.format(np.count_nonzero(seed_img))
    print 'Correct seeds  : {}'.format(np.count_nonzero(correct_seed_img))
    print 'Detected rocks : {}'.format(detect)
    print 'Precision      : {}'.format(round(pre    , 3))
    print 'Recall         : {}'.format(round(recall , 3))
    print 'F-measure      : {}'.format(round(f      , 3))
    print 'Ateempt number : {} / {}'.format(i+1, atp_num)
    print '======================='

def display_result2(img, detect_img, pred_ror, true_ror, pred_seed):

        b,g,r = cv2.split(img)
        r[detect_img==255] = 255
        g[detect_img==150] = 255
        res = cv2.merge((b,g,r))

        b,g,r = cv2.split((img*0.9).astype(np.uint8))
        r[np.array(pred_ror==255) & np.array(true_ror == 255)] = 255 # PandT
        g[np.array(pred_ror==255) & np.array(true_ror == 0)] = 255 # P
        b[np.array(pred_ror!=255) & np.array(true_ror == 255)] = 255 # T
        res2 = cv2.merge((b,g,r))

        b,g,r = cv2.split((img*0.9).astype(np.uint8))
        r[pred_seed !=0] = 255 # PandT
        res3 = cv2.merge((b,g,r))

        cv2.imshow('res', res)
        cv2.imshow('res2',res2)
        cv2.imshow('res3',res3)

        return res

def display_result3(convex, pre2, recall2, param):
    
    f = 2. * recall2 * pre2 / (recall2+pre2)

    print '======================='
    print 'param  = ', param
    print 'convex = ', convex
    print 'pre2   = ', pre2
    print 'recall2= ', recall2
    print 'f      = ', f
    print '======================='

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
        params = [[167, np.pi, 0.12],[167, np.pi*7/4, 0.12]]

    else:
        ## 配列
        # thresh = range(153, 220, 1)
        # sigma = [4,5]
        # bias = np.arange(0.1,0.6,0.1)
        # direction = [np.pi * 0.9, np.pi, np.pi*1.1]
        # dif2 = np.arange(0,0.3,0.03)
        th1s = np.arange(0,0.5,0.02)

        ## 定数
        threshes = [167]
        sigmas = [4]
        biases = [0]
        # directions = [np.pi * 7 / 4]
        directions = [np.pi]

        # th1s = [0,0.02]
        th2s = [0.09]

        params = []
        
        ## 固定したいものを上層に
        for thresh in threshes:
            # for sigma in sigmas:
                # for bias in biases:
                    for direction in directions:
                        for th1 in th1s:
                            # for th2 in th2s:
                                # params.append((int(thresh), int(sigma), bias, direction, th1, th2))
                                params.append((int(thresh), direction, th1))


    return params

def display_details():
    # グラフの描画
    if display == True:
        dis = np.vstack((dis, np.array([precision,recall,f])))
        if (i+1)%variety == 0:
            ax2.plot(dis[:,0], dis[:,1], "-o",label='sigma = {}'.format(param[1]), color='r')
            ax2.legend()
            dis = np.empty((0,3), int)
    
    # rangeのscoreも出すとき
    if detail == True:
        pred_seed_img = rd.main(img, params[best])
        precision, recall, true_features, detect_img = ev.main(pred_seed_img, true_ror, label_img, true_features)
        range_p, range_r = get_detail_score(true_features, pred_seed_img)
        print 'Prc by range   : {}'.format(range_p)
        print 'Rcl by range   : {}'.format(range_r)
    
    # 検出結果を画像で出すとき
    if result_image == True:
        result_best = get_detect_img(img, detect_img)
        result_stock = get_detect_img(img, detect_stock)
        cv2.imshow('best result',result_best)
        cv2.imshow('total result',result_stock)


if __name__ == '__main__':

    # filename = 'spirit006-1.png'
    filename = 'spirit118-1.png'

    img = cv2.imread('../../data/g-t_data/resized/{}'.format(filename))
    true_ror = cv2.imread('../../data/g-t_data/label/{}'.format(filename),0)
    print 'Target image : {}'.format(filename)



    # main processing
    main(img, true_ror)
    cv2.waitKey(-1)