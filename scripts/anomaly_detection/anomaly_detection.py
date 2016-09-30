 # -*- coding: utf-8 -*-
'''
    Test function
    
    Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

def main(features, models):


    model = models[2]
    # plt.hist(features[:,2])
    # plt.show()
    
    ls = [] 
    for f in features[:,2]:
        l,l2 = get_likelihood(f, model) # 各観測データに対する尤度
        print l, l2

        ls.append(l2)

    print 'true',np.argmax(features[:,2]),np.max(features[:,2])
            
    print 'maxsize',np.argmax(ls),features[np.argmax(ls),2]
    print 'minsize',np.argmin(ls),features[np.argmin(ls),2]


    plt.scatter(features[:,2], ls)
    plt.show()




    # size = features[np.argmin(ls),2] 
    size = np.argmin(ls)
    # print np.argmin(ls)
    features[np.argmin(ls),0] = 200000




    return 0,0,0


def get_likelihood(x, model):
    '''
        入力されたモデルに対するxのlikelihoodを求める
        対数尤度にする？
    '''

    p = 1.
    p2 = 0.
    
    for mu, var in zip(model.means_[:,0], model.covars_[:,0,0]):

        p = p * gaussian(x,mu,var)
        p2 += gaussian(x,mu,var)

    return p,p2


def gaussian(x,m,v):
    
    p = math.exp(- pow(x - m, 2) / (2 * v)) / math.sqrt(2 * math.pi * v)
    return p



if __name__ == '__main__':
    main()