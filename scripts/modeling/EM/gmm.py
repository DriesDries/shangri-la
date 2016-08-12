# -*- coding: utf-8 -*-
'''
混合ガウス分布を生成し、その分布に対して
フィッティングを行う
Usage: $ python gmm.py <argv>

''' 
import time
import math
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab


def main():

    '''fitting to the bimodal gaussian distribution '''
    data = get_bgd()
    ema_bgd(data)

def get_bgd():
    '''
        csvファイルから配列を生成して返す
    '''

    sam = open('../data/old_faithful.csv', 'r')
    reader = csv.reader(sam, delimiter=' ')

    # data = np.array([])
    data = []


    for raw in reader:
        data.append([float(raw[0]), float(raw[1])])

    sam.close()

    data = np.array(data)
    data = data[:,0]

    return data

def ema_bgd(data):
    '''
    Calculate parameters based on Expectation–Maximization Algorithm; EMA
    '''
    pi = 0.5 # 負担率
    ms = [random.choice(data), random.choice(data)] # randomでスタートを決める
    vs = [np.var(data), np.var(data)] # dataの分散
    T = 50  #反復回数, iteration number
    ls = []  #対数尤度関数の計算結果を保存しておく

    #結果のプロット
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # 描画方法
    ax1.set_xlim(min(data), max(data))
    ax1.set_xlabel("x")
    ax1.set_ylabel("Probability")
    ax2.set_xlabel("step")
    ax2.set_ylabel("log_likelihood")
    ax2.set_ylim(-500,0)
    ax2.set_xlim(0, T)

    for i in range(T):

        '''EM Algorithm'''
        burden_rates = e_step(xs=data, ms=ms, vs=vs, pi=pi) # Eステップ
        ms, vs, pi = m_step(xs=data, burden_rates=burden_rates) # Mステップ
        ls.append(calc_log_likelihood(data, ms, vs, pi)) # 対数尤度を更新する
        print ls[i]
        # 描画
        xs = np.linspace(min(data), max(data), 200)
        norm1 = mlab.normpdf(xs, ms[0], math.sqrt(vs[0]))
        norm2 = mlab.normpdf(xs, ms[1], math.sqrt(vs[1]))
        ax1.hist(data, 20, normed=1, color='dodgerblue')
        ax1.plot(xs, (1 - pi) * norm1, color="orange", lw=3)
        ax1.plot(xs, pi * norm2, color="orange", lw=3)
        # ax1.plot(xs, (1 - p) * norm1 + p * norm2, color="red", lw=3)

        ax2.plot(np.arange(len(ls)), ls, color='dodgerblue')        

        if i==T-1: # 収束条件
            print i
            ax1.plot(xs, (1 - pi) * norm1 + pi * norm2, color="red", lw=3)
            [ax1.lines.pop(0) for l in range(2)] # remove line 
            print '...Converge in the {}th...'.format(i+1)
            plt.pause(-1)
            break

        # plt.pause(0.1)
        [ax1.lines.pop(0) for i in range(2)] # remove line 

def e_step(xs, ms, vs, pi):
    '''
        現在のms,vsを用いて負担率(burden_rates)を求める。
    dst  : burden_rates -> それぞれのサンプルに対する負担率が入ったlist。len(burden_rates)=len(xs) 
    param: d            -> 負担率の分母。すべてのガウス分布を足し合わせたもの。
           n            -> 負担率の分子。あるガウス分布のみの値。
           burden_rate  -> 負担率
    '''

    burden_rates = []

    for x in xs:
        d = (1 - pi) * gaussian(x, ms[0], vs[0]) + pi * gaussian(x, ms[1], vs[1])
        n = pi * gaussian(x, ms[1], vs[1])
        burden_rate = n / d
        burden_rates.append(burden_rate)

    return burden_rates

def m_step(xs, burden_rates):
    
    '''ガウス分布1(k=1)'''
    n = sum([(1 - r) * x for x, r in zip(xs, burden_rates)])
    d = sum([1 - r for r in burden_rates])
    mu1 = n / d

    n = sum([(1 - r) * pow( x - mu1, 2) for x, r in zip(xs, burden_rates)])
    var1 = n / d
    
    '''ガウス分布2(k=2)'''
    d = sum(burden_rates)
    n = sum([r * x for x, r in zip(xs, burden_rates)])
    mu2 = n / d

    n = sum(r * pow(x - mu2, 2) for x, r in zip(xs, burden_rates))
    var2 = n / d

    ''' pの更新 '''
    N = len(xs)
    pi = sum(burden_rates) / N

    return [mu1, mu2], [var1, var2], pi

def gaussian(x, m, v):
    '''
    ガウシアン分布にパラメータを代入したg(x,m,v)を返す。
    dst : p -> float
    '''
    p = math.exp(- pow(x - m, 2) / (2 * v)) / math.sqrt(2 * math.pi * v)

    return p

def calc_log_likelihood(xs, ms, vs, p):
    s = 0
    for x in xs:
        g1 = gaussian(x, ms[0], vs[0])
        g2 = gaussian(x, ms[1], vs[1])
        s += math.log((1 - p) * g1 + p * g2)
    return s




# '''=========================================================='''
def generate_ugd(mu,var,N):
    '''
        Generate Gaussian distribution
        args : mu, var, N -> 平均,分散,サンプル数
        dst  : data       -> サンプルデータ
    '''

    std = math.sqrt(var)  # 標準偏差
    data = np.random.normal(mu,std,N) # データの入手
    
    return data

def mle_ugd(data):
    '''
    Calculate parameters based on Maximum Likelihood Estimation;MLE

    '''
    x = range(len(data))

    #最尤推定で平均と分散を求める
    mu_ml = np.mean(data)
    var_ml = np.var(data, ddof=1)  #不偏推定量を用いる

    # 描画するための準備
    std_ml = math.sqrt(var_ml) # 標準偏差を求める   
    xs = np.linspace(min(data), max(data), 200) # sampleのxの生成
    norm = mlab.normpdf(xs, mu_ml, std_ml) # 

    # 描画
    plt.plot(xs, norm, color="red")
    plt.hist(data, bins=20, normed=1, color='blue', alpha=0.5)
    plt.pause(-1)

def module2():
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''



if __name__ == '__main__':

    main( )