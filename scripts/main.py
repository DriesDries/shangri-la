# -*- coding: utf-8 -*-
'''
    Usage: $ python template.py <argv>

    ・get_detail_scoreを複数回使う場合は、true_features[:,0]を初期化する
''' 
 
import os, sys, time, copy

import cv2
import numpy as np
import skimage.measure as sk
import matplotlib.pyplot as plt

import rock_detection.rock_detection as rd
import evaluate.rd_eval as ev
import feature_extraction.feature_extraction as fe
import modeling.modeling as md
import anomaly_detection.anomaly_detection as ad


fig = plt.figure(figsize=(16,9))
# ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(111)

# ax1.set_xlim(-2,4)
# ax1.set_ylim(0,1)
# ax1.set_ylabel('precision or recall')
# ax1.set_title('Target selection performance')
# ax1.set_xticks([0,2])
# ax1.set_xticklabels(['Precision','Recall'])

ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_xlabel('precision')
ax2.set_ylabel('recall')
ax2.set_title('Detection performance')


def main(src, true_ror):
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
               featues -> features of true ror
                   - features[:,0]: detected or NONdetected (1 or 0)
                   - features[:,1]: region ID
                   - features[:,2]: y-coordinate
                   - features[:,3]: pixel-size
                   - features[:,4]: color (average of intensitys)
    '''

    img = copy.deepcopy(src)
    
    params, variety = get_params()
    true_ror, label_img, true_features = get_true_features(img, true_ror)

    detail = 1
    display = 1
    result_image = 1

    precisions = []
    recalls = []
    results = np.empty((0,3), int)
    dis     = np.empty((0,3), int)
    detect_stock = np.zeros_like(true_ror)
    detect_stock[true_ror!=0] = 150
    detect_stock_list = np.array(true_features)[:,0].copy()

    for i, param in enumerate(params):

        ## Rock detection and evaluation
        pred_ror, pred_seed_img = rd.main(img, param)
        precision, recall, true_features, detect_img = ev.main(pred_seed_img, true_ror, label_img, true_features)
        f = 2.*recall*precision / (recall+precision)

        print '======================='
        print 'Seed threshold : {}'.format(int(param[0]))
        print 'Gabor sigma    : {}'.format(param[1])
        print 'Gabor bias     : {}'.format(param[2])
        print 'Sun direction  : {}π'.format(param[3]/np.pi)
        print 'Rock quantity  : {}'.format(np.amax(label_img))
        print 'Seed quantity  : {}'.format(len(np.nonzero(pred_seed_img)[0]))
        print 'Correct seeds  : {}'.format(true_features[:,0].sum())
        print 'Detected rocks : {}'.format((np.array(np.nonzero(true_features[:,0]))).shape[1])
        print 'Precision      : {}'.format(round(precision, 3))
        print 'Recall         : {}'.format(round(recall   , 3))
        print 'F-measure      : {}'.format(round(f        , 3))
        print 'Ateempt number : {} / {}'.format(i+1,params.shape[0])
        print '======================='

        results = np.vstack((results, np.array([precision,recall,f])))

        ## Initialization and renew
        detect_stock_list += true_features[:,0]
        detect_stock[detect_img == 255] = 255
        true_features[:,0] = 0

        # グラフの描画
        if display == True:
            dis = np.vstack((dis, np.array([precision,recall,f])))
            if (i+1)%variety == 0:
                ax2.plot(dis[:,0], dis[:,1], "-o",label='sigma = {}'.format(param[1]))
                ax2.legend()
                dis = np.empty((0,3), int)
        
    # best scoreの表示
    best = np.argmax(results[:,2])
    print '***** BEST SCORE ******'
    print 'Seed threshold : {}'.format(int(params[best][0]))
    print 'Gabor sigma    : {}'.format(params[best][1])
    print 'Gabor bias     : {}'.format(params[best][2])
    print 'Sun direction  : {}π'.format(params[best][3]/np.pi)
    print 'Precision      : {}'.format(round(results[best,0], 3))
    print 'Recall         : {}'.format(round(results[best,1], 3))
    print 'F-measure      : {}'.format(round(results[best,2], 3))
    print 'Ateempt number : {} / {}'.format(best+1,params.shape[0])
    print '***********************'
    
    # rangeのscoreも出すとき
    if detail == True:
        pred_ror, pred_seed_img = rd.main(img, params[best])
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

def get_detect_img(img,detect_img):
    '''
    '''
    b,g,r = cv2.split((img * 0.8).astype(np.uint8))

    r[detect_img==255]=255
    g[detect_img==150]=255

    result = cv2.merge((b,g,r))

    return result


def get_detail_score(true_features, seed_img):
    '''
       近い -> 遠い
    '''
    
    true_ranges = true_features[:,2]
    true_psizes = true_features[:,3]
    correct_seeds = true_features[:,0]

    detected_features = true_features[np.nonzero(true_features[:,0])]
    detected_ranges = detected_features[:,2]
    detected_psizes = detected_features[:,3]
    detected_seeds = true_features[:,0].sum()

    all_seed_list = np.nonzero(seed_img)[1]


    '''by Range'''
    ## Recall 範囲内の全岩の数
    ar =  (300 < true_ranges).sum()                           # close
    br = ((200 < true_ranges) == (true_ranges <= 300)).sum()
    cr = ((100 < true_ranges) == (true_ranges <= 200)).sum()
    dr =                         (true_ranges <= 100 ).sum()  # far

    ## Detected 検出した岩の数
    ad =  (300 < detected_ranges).sum()                           # close
    bd = ((200 < detected_ranges) == (detected_ranges <= 300)).sum()
    cd = ((100 < detected_ranges) == (detected_ranges <= 200)).sum()
    dd =                             (detected_ranges <= 100).sum()  # far

    ## Precision 指定した範囲の全種の数
    ap = len(all_seed_list[300 < all_seed_list])
    bp = len(all_seed_list[(200 < all_seed_list) == (all_seed_list <= 300)])
    cp = len(all_seed_list[(100 < all_seed_list) == (all_seed_list <= 200)])
    dp = len(all_seed_list[                         (all_seed_list <= 100)])

    ## Precision 正しかった種の数
    ads = correct_seeds[300 < true_ranges].sum()
    bds = correct_seeds[(200 < true_ranges) == (true_ranges <= 300)].sum()
    cds = correct_seeds[(100 < true_ranges) == (true_ranges <= 200)].sum()
    dds = correct_seeds[                       (true_ranges <= 100)].sum()

    range_recall = [1.*ad/ar, 1.*bd/br, 1.*cd/cr, 1.*dd/dr]
    range_precision = [1.*ads/ap, 1.*bds/bp, 1.*cds/cp, 1.*dds/dp]

    '''by Size'''
    ## True
    er =  (300 < true_psizes).sum()                           # large
    fr = ((200 < true_psizes) == (true_psizes <= 300)).sum()
    gr = ((100 < true_psizes) == (true_psizes <= 200)).sum()
    hr =                          (true_psizes <= 100).sum()  # small

    ## Detected
    ed =  (300 < detected_psizes).sum()                           # large
    fd = ((200 < detected_psizes) == (detected_psizes <= 300)).sum()
    gd = ((100 < detected_psizes) == (detected_psizes <= 200)).sum()
    hd =                             (detected_psizes <= 100).sum()  # small

    size_recall = [1.*ed/er, 1.*fd/fr, 1.*gd/gr, 1.*hd/hr]

    ## Size precision
    ap = len(all_seed_list[300 < all_seed_list])
    bp = len(all_seed_list[(200 < all_seed_list) == (all_seed_list <= 300)])
    cp = len(all_seed_list[(100 < all_seed_list) == (all_seed_list <= 200)])
    dp = len(all_seed_list[                         (all_seed_list <= 100)])

    return range_precision, range_recall

def get_params():

    thresh = range(140,191,5)
    sigma = [4,5]
    bias = np.arange(0.1,0.6,0.1)
    direction = [np.pi * 0.9, np.pi, np.pi*1.1]

    # もし定数にするとき
    # thresh = [140]
    # sigma = [4]
    # bias = []
    direction = [np.pi]

    params = []

    # 固定したいものを上層に
    for s in sigma:
        for th in thresh:
            for b in bias:
                for d in direction:
                    params.append((int(th), int(s), b, d))

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

if __name__ == '__main__':

    img = cv2.imread('../../data/g-t_data/resized/spirit118-1.png')
    label_img = cv2.imread('../../data/g-t_data/label/spirit118-1.png')
    true_ror = get_binary_ror(label_img)

    # main processing
    main(img, true_ror)

    plt.pause(-1)
    cv2.waitKey(-1)
    

    # features = fe.main(src, s_ror) ; print 'fe'
    # model = md.main(features)      ; print 'modeling'
    # target = ad.main(model)        ; print 'ad'

