# -*- coding: utf-8 -*-
'''
入力されたデータを指定した確率分布に基づいてモデル化(フィッティングする)

Usage: $ python modeling.py
''' 
import cv2
import os
import sys

import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy import stats


def poisson(data):
    '''
    入力されたdataに対してポアソン分布を最尤推定法によってあてはめる
    args: data ->一次元配列
    '''
    # ヒストグラムと確率分布の表示
    ax1 = fig.add_subplot(121)
    parameters = np.arange(2,6,1)
    make_graph(data, parameters)


    '''最尤推定法,多分ちがう''' # lmdが変化したときの対数尤度を求め描画し、最大尤度を求める
    # pfm -> 確率質量関数
    ax2 = fig.add_subplot(122)
    lmds = np.arange(2,6,0.01)
    logL = np.array([sum(stats.poisson.logpmf(data, lmd)) for lmd in lmds]) # 対数尤度
    
    ax2.plot(lmds,logL)
    ax2.axvline(x=lmds[np.argmax(logL)], color = 'r') # 最大尤度に線を引く
    
    max_logL = np.amax(logL) # 最大尤度
    max_lmd = lmds[np.argmax(logL)] # 最大尤度のときのパラメータλ

    # グラフの設定
    ax2.set_title('max lambda = %s, max logL = %s'%(max_lmd,int(max_logL)))
    ax2.set_xlabel('lambda')
    ax2.set_ylabel('logL')

    # print 'max_logL =', max_logL,'lambda =', max_lmd # 最大尤度
    return  max_logL, max_lmd
    
def make_graph(data, parameters):
    '''
    入力されたdataのhistgramと、確率分布をグラフとして表示する
    '''
    '''dataのhistgram'''
    plt.hist(data, bins=np.arange(-0.5, 8.5, 1.0), color='w')


    '''gaussian distribution'''
    x = np.arange(-10,10,0.1)
    gau = 30 * stats.norm.pdf(x-3) # もっとあるはず
    plt.plot(x, gau, color='g')

    
    '''poisson distribution''' # poisson分布はyが整数値でしか値を取らない
    y = np.arange(0,10,1)
    for lmd in parameters:
        psn = 50 * pandas.Series(stats.poisson.pmf(y,lmd), index=y)
        # plt.plot(psn, '--o', label=lmd)

    # グラフの設定
    plt.legend(loc='upper right', title='lambda')
    plt.title('Distribution and Histgram')
    plt.xlabel('y')
    plt.ylabel('prob')
    # plt.xlim([-10,10])
    # plt.ylim([0,15])

def analysis(data):
    '''
    入力された一次元配列の情報を表示する
    pandas.Seriesはpandasのデータ構造
        - 出力した際に順序も同時に表示される
        - strも格納できる
        - describeとかの関数が使える
    '''

    data = pandas.Series(data) 
    print data.describe()
    print data.value_counts(sort = False)
    print data.var(), np.sqrt(data.var())
    print data


if __name__ == '__main__':

    # read class
    fig = plt.figure(figsize=(16,9))
    
    data = [2, 2, 4, 6, 4, 5, 2, 3, 1, 2, 0, 4, 3, 3, 3, 3, 4, 2, 7, 2, 4, 3, 3, 3, 4, 3, 7, 5, 3, 1, 7, 6, 4, 6, 5, 2, 4, 7, 2, 2, 6, 2, 4, 5, 4, 5, 1, 3, 2, 3]


    # for i,val in enumerate(data):
        # if val == 2:
            # data[i] = 5
    
    # make_graph(data)
    poisson(data)
    plt.pause(-1)
