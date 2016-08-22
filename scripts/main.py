# -*- coding: utf-8 -*-
'''
Usage: $ python template.py <argv>
''' 
 
import os, sys, time, copy

import cv2
import numpy as np

import rock_detection.merge as merge
import rock_detection.small_rock_detection as srd
import rock_detection.large_rock_detection as lrd
import rock_detection.tm_eval as rd
import feature_extraction.feature_extraction as fe
import modeling.modeling as md
import anomaly_detection.anomaly_detection as ad



def main(src):
    '''
        args : src     ->  3ch-image
        dst  : model   ->  model
               target  ->  taeget
        param:         -> 
    '''
    img = copy.deepcopy(src)

    s_ror = srd.main(img,np.pi)    ; print 'srd'
    # l_ror = lrd.main(img)          ; print 'lrd'
    # ror = merge.main(s_ror,l_ror)  ; print 'merge'

    # features = fe.main(src, s_ror) ; print 'fe'
    # model = md.main(features)      ; print 'modeling'
    # target = ad.main(model)        ; print 'ad'

    cv2.imshow('s_ror',s_ror) 
    # cv2.imshow('l_ror',l_ror)
    # cv2.imshow('ror',ror)

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

    ''' fill '''
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


if __name__ == '__main__':

    img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png')

    main(img)

    cv2.waitKey(-1)
    

