# -*- coding: utf-8 -*-
'''
ある確率分布にしたがって乱数を生成し、その乱数に対して曲線フィッティングを行う
Usage: $ python curve_fitting.py
''' 

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def main():

    # sin関数の生成
    x_real, y_real = generate_sin()

    # ノイズを加えたsin関数の生成
    N = 100
    x_train, y_train = random_number(x_real, y_real, N)

    # フィッティング
    M = 3 # 次数
    W = fitting(x_train, y_train, M, x_real, y_real)

    # フィッテイングしたパラメータで描画する
    y_estimate = [ Y(x, W, M) for x in x_real]

    # plot
    plt.plot(x_real,  y_real, '--', label='sin') # 目的関数
    plt.plot(x_train, y_train, 'bo',  label='train') # 訓練データ
    plt.plot(x_real,  y_estimate, 'r',label='estimateion', linewidth = 2) # フィッティングした関数
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(-2,2)
    plt.pause(-1)

def generate_sin():
    # Sin curve
    x_real = np.arange(0, 1, 0.01)
    y_real = np.sin(2 * np.pi * x_real) # 周期2π

    return x_real, y_real

def random_number(x_real, y_real, N):
    '''
        データの生成
        args : N -> 訓練データの数
        dst  :      -> 
        param:      -> 
    '''
    # Training data
    x_train = np.arange(0,1,1./N)

    loc = 0
    scale = 0.3
    y_train = np.sin(2 * np.pi * x_train) + np.random.normal(loc, scale, N)

    return x_train, y_train

def Y(x, W, M):
    '''
        Y(x,W) = w_0 + w_1*x + ... + w_M*x^M = sigma_{j=1 ~ M} w_j*x^M
        に代入する
    '''

    Yi = np.array([ W[i] * (x ** i) for i in range(M+1) ])
    Y = Yi.sum()
    return Y

def fitting(x, y, M, x_real, y_real):
    '''
        入力されたx,yにフィッティングする
        args :      -> 
        dst  :      -> 
        param: M -> フィッティング関数の次数
    '''
    # パラメータを求める
    A = np.zeros((M+1,M+1))

    for i in range(M+1):
        for j in range(M+1):
            A[i,j] = (x**(i+j)).sum()

    T = np.array([ ((x**i)*y).sum() for i in range(M+1) ])
    W = np.linalg.solve(A, T) # Ainv と T の積
    
    return W


if __name__ == '__main__':
    
    main()


