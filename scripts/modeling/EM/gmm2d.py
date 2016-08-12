# -*- coding: utf-8 -*-
'''
二次元の特徴量に対して、混合ガウス分布をモデリングする
Usage: $ python gmm2.py <argv>
・結果を重ねて表示

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

N = 5
EM = mixture.GMM(n_components=N, covariance_type='full',n_iter=100)

def main():

    '''fitting to Gaussian Mixture Model '''
    data = get_data(dim=2)
    em(data, N=5)

def get_data(dim):
    '''
        csvファイルから配列を生成して返す
    '''

    sam = open('../data/old_faithful.csv', 'r')
    reader = csv.reader(sam, delimiter=' ')

    data = []

    for raw in reader:
        data.append([float(raw[0]), float(raw[1])])

    sam.close()

    data = np.array(data)

    if dim == 1:
        data = data[:,0] # データを1次元にする

    return data

def em(data,N):
    '''
        混合ガウス分布を用いてdataをモデリングする
        args :      -> 
        dst  :      -> 
        param: N -> いくつのガウシアン分布を用いるか 
    '''
    # fitting
    EM.fit(data)

    # 準備
    x = np.linspace(min(data[:,0]), max(data[:,0]), 1000)
    y = np.linspace(min(data[:,1]), max(data[:,1]), 1000)
    X, Y = np.meshgrid(x, y)
    
    print 'n_components = {}'.format(N)
    # print EM.weights_ # 混合係数, 足したら1になる
    # print EM.converged_ # 収束してればTrue
    # print EM.means_  # ガウス分布の平均(頂点の座標になる)、2変量ガウス分布だから2次元
    # print EM.covars_ # 分散
    ZS = np.zeros_like(X)
    Z = []
    # 結果の描画
    for k in range(N):

        # ガウス分布の等高線を描画
        Z.append(mlab.bivariate_normal(X, Y, np.sqrt(EM.covars_[k][0][0]), 
                                        np.sqrt(EM.covars_[k][1][1]),
                                        EM.means_[k][0], EM.means_[k][1],
                                        EM.covars_[k][0][1]))
        ZS = ZS + Z[k]

    display_3D(X, Y, ZS)
    histgram_3D(data)
    # display_contour(X,Y,Z)
    
    plt.pause(-1)
    

    # likelihoods_em.append(EM.score(data).mean()) # fitした結果を受け取ってる
    # print likelihoods_em
    # plt.scatter(i,likelihoods_em)
    # plt.pause(1)




    # plt.hist(data, bins=20, normed=1, color='blue', alpha=0.5)

def histgram_3D(data):
    '''
    入力された二次元配列を3Dhistgramとして表示する
    '''

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    x = data[:,0]
    y = data[:,1]

    hist, xedges, yedges = np.histogram2d(x, y, bins=30)
    X, Y = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)

    # bar3dでは行にする
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros(len(X))

    # 表示するバーの太さ
    dx = (xedges[1] - xedges[0]) * np.ones_like(Z)
    dy = (yedges[1] - yedges[0]) * np.ones_like(Z)
    dz = hist.flatten() # これはそのままでok

    # 描画
    ax.bar3d(X, Y, Z, dx, dy, dz, color='b', zsort='average')


def display_3D(X,Y,Z):
    '''
        入力画像を3Dで表示する
        args: X,Y,Z
        dst : None
    '''
    # データの準備
    from mpl_toolkits.mplot3d import Axes3D

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)

    # 設定
    ax.set_xlabel('pixel')
    ax.set_ylabel('pixel')        
    ax.set_zlabel('intensity')
    # ax.set_zlim(0, 300)
    ax.set_title('Image')
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap = 'jet',linewidth=0)
    # ax.plot_wireframe(X,Y,Z, cmap = 'Greys', rstride=30, cstride=30)

    # plt.pause(-1) # これだけでok

def display_contour(X,Y,Z):
    '''
        等高線を表示
    '''
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = EM.score_samples(XX)[0]
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z)
    CB = plt.colorbar(CS)


if __name__ == '__main__':

    main( )