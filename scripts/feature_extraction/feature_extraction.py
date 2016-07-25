# -*- coding: utf-8 -*-
'''
入力画像の特徴抽出を行って特徴の一覧を返すスクリプト
Usage: $ python template.py <argv>
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
        dst  : size, shape, composition     -> 
        param:      -> 
    '''
    ror = copy.deepcopy(ror_ori)
    
    # 組成特徴の抽出
    compositons = get_composition(src,ror) # compositions[i]に一つのが入る

    # それぞれの領域に楕円をfittingし、形状と大きさの特徴を抽出する
    ellipses = fit_ellipse(src,ror) # 中には楕円のパラメータが入ってる
    print 'ellipse number = ',len(ellipses)
    centers, sizes, shapes = get_parameter(ror,ellipses)

    # 画像上の位置に基づいてパラメータを修正する
    modify_parameter(centers,sizes,shapes)

    # 特徴の結合
    features = []

    # 結果の表示
    display_result(features)

    # 結果の表示 ###################################################
    # cv2.imshow('input',src)
    # cv2.drawContours(ror,contours,-1,(0,255,0),3)
    # cv2.drawContours(con,contours,-1,color=(0,255,0),thickness=1)
    # cv2.imshow('ror',ror)
    # cv2.imshow('con',con)
    ##############################################################

    return features

def get_composition(src,ror):
    '''
        入力されたrorをimgにmaskしてそれぞれの領域の色の平均をとる
    '''        

    # rorの領域に番号つける,# 非岩領域も算出されてるからnumは一つ多く出る
    labels, num = sk.label(ror, return_num = True) 
    compositions = []

    for i in range(1,num,1): # 非岩領域以外で
        
        img = copy.deepcopy(src) # 初期化
        img[labels!=i]=0 # ror以外は0とする

        # 領域の平均値を組成の値とする
        composition = img.sum() / len(np.nonzero(img)[0])
        compositions.append(composition)

    return compositions

def fit_ellipse(img, ror):
    '''
        入力画像に輪郭を描画して返す
        args : img -> 元画像
               contours -> 輪郭が入った配列.contours[i]が一つの輪郭を表す
        dst  : ellipses -> 楕円fittingしたパラメータ。ellipses[i]がcontours[i]の輪郭にfitした楕円
        param:      -> 
    '''
    
    # 輪郭の抽出, contours[i]にひとつづつ輪郭の座標が入る
    contours, hierarchy = cv2.findContours(ror, 1, 2)
    
    ellipses = []
    
    for i,contour in enumerate(contours):
        ell = np.zeros_like(img)
        cv2.drawContours(ell,contour,-1,(255,255,255),2)

        if len(contour)>5: # 楕円フィッティング
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(ell,ellipse,(0,0,255),1)
            ellipses.append(ellipse)

        else: # 小さすぎると楕円フィッティングできない
            continue

        # 結果の表示
        # cv2.imshow('con',ell)
        # cv2.waitKey(0)
    
    return ellipses

def get_parameter(img, ellipses):
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
        
        centers.append(ellipse[0])                  # 楕円の中心座標
        shapes.append(ellipse[1][0]/ellipse[1][1])  # 長径と短径の比
        sizes.append(ellipse[1][0] + ellipse[1][1]) # 長径と短径の和

    return centers, sizes, shapes

def modify_parameter(centers, sizes, shapes):
    pass

def display_result(features):
    pass



if __name__ == '__main__':

    # img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')
    # img = img[400:800,400:800]        
    
    # sample image
    img = np.zeros((400,400,3)).astype(np.uint8)
    cv2.ellipse(img,((100,100),(30,60),90),255,-1) # 楕円の描画
    cv2.ellipse(img,((200,200),(30,90),60),200,-1)
    cv2.ellipse(img,((300,300),(30,60),90),150,-1)
    ror = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 岩領域の用意

    cv2.waitKey(-1)