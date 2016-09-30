#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    ============================
    Principal Component Analysis
    ============================

    入力データを主成分分析(Principal Component Analysis; PCA)する．
    特異値分解によるPCA
"""
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

def pca(data, labels, N):

    pca = PCA(n_components=N)
    X_pca = pca.fit_transform(data)

    ## show attribute
    print pca.components_
    # print pca.explained_variance_ # 寄与値
    print pca.explained_variance_ratio_ # 寄与率
    print pca.mean_
    print pca.noise_variance_


    ## plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap=plt.cm.Paired)
    ax.set_title('First three PCA directions')
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvector')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()

if __name__ == '__main__':
    
    # import some data
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target

    pca(data, labels, N = 3)