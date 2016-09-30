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
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


import rock_detection.rock_detection as rd

# ta = rd.TextureAnalysis()
# fig = plt.figure(figsize=(16,9))
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)

def projection():
    
    fig = plt.figure(figsize=(16,9))
    ax = Axes3D(fig)

    H = 1.4
    FOV = 40.
    P = 100.
    pan = 60.
    
    x = range(0,int(P))
    y = range(0,int(P))
    X,Y = np.meshgrid(x,y)
    Y2 = H  * np.tan( ( 1.*Y*FOV/P + pan - FOV/2 ) *np.pi/180)
    X2 = Y2 * ( np.tan(abs(  FOV * X/P    ) *np.pi/180) - np.tan((FOV/2) *np.pi/180) )
    # np.tan(1.*row*fov/rows + pan - 1.*fov/2) / np.tan(pan)
    
    ax.scatter(X2, Y2 , 1, s=3, c='b', cmap='gray', linewidths=0)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    print np.min(Y2), np.max(Y2)

        
    plt.show()

def texture(img, responses):
    
    center = []
    dis = np.zeros_like(img[:,:,0]).astype(np.float64)
    
    # Center Acquitision    
    for n in range(responses.shape[0]):
        res = responses[n,:,:].sum() / 400**2
        center.append(res)

    ## var processing
    for j in range(dis.shape[0]):
        for i in range(dis.shape[1]):
            dis[j,i] = np.linalg.norm(responses[:,j,i] - center[:])

    dis = dis / np.max(dis)
    # img2 = (img*0.8).astype(np.uint8)
    
    # # display result
    # for v in np.arange(0, np.max(dis)+0.02, 0.01):
    #     b,g,r = cv2.split(img2)
    #     v1 = np.array(v<=dis)
    #     v2 = np.array(dis<=(v+0.05))
    #     r[v1 * v2] = 255
    #     res = cv2.merge((b,g,r))
    
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # a = ax2.imshow(dis)
    # fig.colorbar(a, cax=cax)

    return dis, center

def intensity(img):

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center = gimg.sum() / 400**2

    dis = np.zeros_like(img[:,:,0]).astype(np.float64)
    for j in range(dis.shape[0]):
        for i in range(dis.shape[1]):
            dis[j,i] = abs(gimg[j,i] - center)
    dis = dis / np.max(dis)
    
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # a = ax1.imshow(dis)
    # fig.colorbar(a, cax=cax)

    return dis

def var(tvar, ivar):
    
    var = np.zeros_like(tvar)
    for j in range(tvar.shape[0]):
        for i in range(tvar.shape[1]):
            var[j,i] = np.sqrt(tvar[j,i]**2 + ivar[j,i]**2)

    return var

if __name__ == '__main__':


    projection()

    # filename = 'spirit006-1.png'
    filename = 'spirit118-1.png'

    img = cv2.imread('../../data/g-t_data/resized/{}'.format(filename))
    true_ror = cv2.imread('../../data/g-t_data/label/{}'.format(filename),0)
    print 'Target image : {}'.format(filename)

    responses = ta.filtering(img, name='MR', radius=3)

    plt.figure()
    for res in responses:
        plt.imshow(abs(res-0.5))
        plt.show()


    ## main preocessing
    tvar, tcenter = texture(img, responses) ## 既に画素値の変化も入ってる
    plt.figure()
    plt.imshow(tvar)
    plt.pause(-1)
    # ivar = intensity(img)
    # var = var(tvar, ivar)

    # 大きい岩とsoilで
    tvar2 = tvar.copy()
    tvar3 = tvar.copy()

    tvar2[true_ror!=255] = 0 # rock
    tvar3[true_ror==255] = 0 # soil

    ## 0は抜いて入れる
    ax1.imshow(tvar2)
    ax3.imshow(tvar3)

    tvar2 = tvar2.flatten()
    tvar3 = tvar3.flatten()

    ax2.hist(tvar2[np.nonzero(tvar2)])
    ax4.hist(tvar3[np.nonzero(tvar3)])
    ax2.set_ylim(0,50000)
    ax4.set_ylim(0,50000)





    plt.show()






    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # a = ax3.imshow(tvar)
    # fig.colorbar(a, cax=cax)
    # ax4.imshow(img)
    # plt.show()