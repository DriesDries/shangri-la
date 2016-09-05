# -*- coding: utf-8 -*-
'''
    Usage: $ python seed_eval.py

''' 
 
import cv2
import numpy as np
import skimage.measure as sk
import matplotlib.pyplot as plt

def main(seed_img, true_ror):
    '''
        args : seed_img -> 1ch, predicted seeds image
               true_ror -> 1ch, ground-truth binary image,
    '''
    ## Get scores
    labels = sk.label(true_ror, return_num = False, background=0)
    pre, recall, pre_count, recall_count, detect_img  = get_score(seed_img, labels)
    f = 2. * recall * pre / (recall+pre)

    return pre, recall, f, detect_img

def get_score(seed_img, labels):

    recall, recall_count, detect_img = get_recall(seed_img, labels)
    precision, pre_count = get_precision(seed_img, labels)
    
    return precision, recall, pre_count, recall_count, detect_img

def get_recall(seed_img, labels):
    '''
        recall = (検出できた岩の数) / (全ての岩の数)
        それぞれの岩領域の中にひとつでも種があれば、検出できたこととする。
        detect_img -> 検出できた岩は255、できなかった岩は150とした画像
    '''
    detect_count = 0
    detect_img = np.zeros_like(seed_img)

    for i in range(0, np.amax(labels)+1):
        
        if np.count_nonzero(seed_img[labels==i]) != 0: # Detected
            detect_count += 1
            detect_img[labels==i] = 255

        else: ## CANNOT Detected
            detect_img[labels==i] = 150

    recall = 1. * detect_count/np.amax(labels)

    return recall, detect_count, detect_img

def get_precision(seed_img, labels):
    '''
        precisionを求める
        それぞれの種が岩領域の中に入っているかどうか
        
        Precision = (岩だった数)/(検出した岩の数)
        
        ror_true -> 岩領域の二値画像
        num      -> 全部の岩の数
    '''
    correct_count = 0
    seed_list = np.nonzero(seed_img)

    for y,x in zip(seed_list[0], seed_list[1]):

        if labels[y,x] >= 0:
            correct_count += 1

    precision = 1. * correct_count / len(seed_list[0])

    return precision, correct_count

if __name__ == '__main__':

    ## Sample image
    seed_img = cv2.imread('../../../data/seed_img.png')
    true_ror = cv2.imread('../../../data/g-t_data/label/spirit118-1.png',0)

    # main processing
    main(seed_img, true_ror)
