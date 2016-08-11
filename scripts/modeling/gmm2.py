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
from sklearn import mixture


def main():

    '''単峰型のガウシアン分布 the unimodal gaussian distribution '''
    # data = generate_ugd(mu=3.0, var=2.0, N=10000)
    # mle_ugd(data)

    '''fitting to the bimodal gaussian distribution '''
    data = get_bgd()
    ema(data)

def get_bgd():
    '''
        csvファイルから配列を生成して返す
    '''

    sam = open('./old_faithful.csv', 'r')
    reader = csv.reader(sam, delimiter=' ')

    # data = np.array([])
    data = []


    for raw in reader:
        data.append([float(raw[0]), float(raw[1])])

    sam.close()

    data = np.array(data)
    # data = data[:,0]

    return data

def ema(data):
    '''
        処理の概要
        args :      -> 
        dst  :      -> 
        param:      -> 
    '''
    likelihoods_em = []
    n = 2
    gmm_em = mixture.GMM(n_components=n, covariance_type='full')

    gmm_em.fit(data)

    x = np.linspace(min(data[:,0]), max(data[:,0]), 1000)
    y = np.linspace(min(data[:,1]), max(data[:,1]), 1000)
    X, Y = np.meshgrid(x, y)
    
    for k in range(n):
        # それぞれのクラスタの平均を描画
        plt.plot(gmm_em.means_[k][0], gmm_em.means_[k][1], 'ro')
        
        # ガウス分布の等高線を描画
        Z = mlab.bivariate_normal(X, Y, np.sqrt(gmm_em.covars_[k][0][0]), 
                                        np.sqrt(gmm_em.covars_[k][1][1]),
                                        gmm_em.means_[k][0], gmm_em.means_[k][1],
                                        gmm_em.covars_[k][0][1])
        plt.contour(X, Y, Z)

    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = gmm_em.score_samples(XX)[0]
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z)
    CB = plt.colorbar(CS)




    plt.scatter(data[:,0],data[:,1])
    plt.pause(-1)
    

    # likelihoods_em.append(gmm_em.score(data).mean()) # fitした結果を受け取ってる
    # print likelihoods_em
    # plt.scatter(i,likelihoods_em)
    # plt.pause(1)




    # plt.hist(data, bins=20, normed=1, color='blue', alpha=0.5)


if __name__ == '__main__':

    main( )