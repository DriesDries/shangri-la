# -*- coding: utf-8 -*-
'''
一次元の特徴量に対して、EM法(Expectation–Maximization Algorithm)
を用いて、混合ガウス分布をモデリングする
Usage: $ python gmm1d.py <argv>

・繰り返し処理が見れるようにする -> 無理っぽい？
・対数尤度を求める -> なんか値がおかしい？
・システムに統合する
・covariance_typeを決める
・オンラインEMアルゴリズムを実装する
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


def main(data,N):
    em(data, N=10)    

def em(data,N):
    '''
        混合ガウス分布を用いてdataをモデリングする
        args : data    -> モデリングしたいデータ
               N       -> いくつのガウス分布の線形重ね合わせで表現するか
        dst  : score   -> 対数尤度(log likelihood)
        param: N -> いくつのガウシアン分布を用いるか 
               EM.weights_   -> 混合係数, 足したら1になる
               EM.covars_    -> それぞれのガウス分布の分散
               EM.means_     -> ガウス分布の平均(頂点の座標になる)、2変量ガウス分布だから2次元
               EM.converged_ -> 収束してればTrue
    '''
    
    data[:,1] = 0

    # fitting
    EM = mixture.GMM(n_components=N, covariance_type='full',n_iter=100,verbose=0)
    EM.fit(data)
    
    # 描画の準備
    x = np.linspace(start=min(data[:,0]), stop=max(data[:,0]), num=1000)
    y = 0
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)


    ps = range(N)
    p = 0

    for k in range(N): # それぞれのガウス分布を描画
        ps[k] = EM.weights_[k] * mlab.normpdf(x, EM.means_[k,0], math.sqrt(EM.covars_[k][0][0]))
        p += ps[k]
        ax1.plot(x, ps[k], color='orange')

    if EM.converged_ == True: # 収束してたら描画
        print '...Converge...'
        ax1.plot(x,p,color='red',linewidth=3)
        ax1.hist(data[:,0], bins = 30, color='dodgerblue', normed=True)

    else: # 収束しなかった場合
        print '!!!Cannot converge!!!'
        
    return EM.score(data).sum()


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

def calc_log_likelihood(xs, ms, vs, p):
    
    s = 0
    for x in xs:
        g0 = gaussian(x, ms[0], vs[0])
        g1 = gaussian(x, ms[1], vs[1])
        # g2 = gaussian(x, ms[2], vs[2])
        # g3 = gaussian(x, ms[3], vs[3])
        # g4 = gaussian(x, ms[4], vs[4])
        # s += math.log(p[0] * g0 + p[1] * g1 + p[2] * g2 + p[3] * g3 + p[4] * g4)
        s += math.log(p[0] * g0 + p[1] * g1)

    return s

def gaussian(x, m, v):
    '''
    ガウシアン分布にパラメータを代入したg(x,m,v)を返す。
    dst : p -> float
    '''
    p = math.exp(- pow(x - m, 2) / (2 * v)) / math.sqrt(2 * math.pi * v)

    return p 

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


if __name__ == '__main__':

    data = get_data(dim=2)
    start = time.time()
    main(data=data, N=5)
    print time.time()-start
    plt.show()