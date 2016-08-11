# coding:utf-8
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.data import astronaut
import matplotlib.pyplot as plt
import math
import copy
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D

def display_3D(img):
    '''
        入力画像を3Dで表示する
        args: 1ch image
        同じwindowで更新したいときは、fig = plt.figure()を#して、呼び出す前の箇所に記述する
    '''
    # データの準備
    x = np.arange(0, len(img[0]), 1)
    y = np.arange(0, len(img[1]), 1)
    X, Y = np.meshgrid(x, y) 
    Z = img


    # plot
    fig = plt.figure() # これを#する
    ax = Axes3D(fig)
    # ax.plot(X, Y)

    # 設定
    fig.canvas.set_window_title('Test')
    ax.set_xlabel('pixel')
    ax.set_ylabel('pixel')        
    ax.set_zlabel('intensity')
    # ax.set_zlim(0, 300)
    ax.set_title('Image')

    # 表示方法 wire or surface
    # ax.plot_wireframe(X,Y,Z)
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap = 'jet',linewidth=0)
    
    # 表示(.s)[sec]
    # plt.pause(.001) # これだけでok
    plt.pause(-1) # これだけでok




# scales = np.array(range(5,25,2))

# kernels = map(lambda x:cv2.getGaborKernel(ksize = (x,x), sigma = 5,theta = np.pi, lambd = x/2, gamma = 10, psi = np.pi * 1/2), scales)

img = cv2.imread('../../image/sample/sample.png')
img = cv2.resize(img,(512,512))
kernel = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)).astype(np.float)
# cv2.imshow('img',img)
img = img.astype(np.float)

for i in range(512):
    for j in range(512):
        
        b = img[j,i][0]
        g = img[j,i][1]
        r = img[j,i][2]
        
        #全て黒のは低く
        if b == 0 and g == 0 and r == 0:
            kernel[j,i]=-100
        
        #緑だけ強いのは高く
        if g>250 and b<30:
            kernel[j,i]=100
        
        #白は高く
        if g>250 and b>250:
            kernel[j,i]=100
        
        #赤は低く
        if r>180 and b<30:
            kernel[j,i]=-100


# img = cv2.imread('../../image/rock/spiritsol118navcam.jpg')
# img = img[400:800,400:800]
# cv2.imshow('img',img)

img = cv2.imread('../../../Dropbox/g-t_data/original/spirit118.png')
img2 = cv2.imread('../../../Dropbox/g-t_data/label/spirit118.png')

dif= img-img2
dif[0:200,0:399]=0
cv2.imshow('dif',dif)
print np.amax(dif)


# print kernel.dtype,np.amax(kernel),np.amin(kernel)
# kernel = kernel + 100
# kernel = kernel.astype(np.uint8)
# print kernel.dtype,np.amax(kernel),np.amin(kernel)
# cv2.imshow('kernel',kernel)
# display_3D(kernel)








# fig = plt.figure()
# for i,kernel in enumerate(kernels):
#  if i == 9:
#     kernel = cv2.normalize(kernel, -1, 1, norm_type = cv2.NORM_MINMAX)
#     print np.amax(kernel),np.amin(kernel)
#     display_3D(kernel)
#     kernel = cv2.normalize(kernel, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
#     cv2.imshow('kernels',kernels[1])
    # cv2.waitKey(-1)

# img = cv2.imread('../image/rock/spiritsol118navcam.jpg',0)
# img = img[400:800,400:800]

cv2.waitKey(-1)