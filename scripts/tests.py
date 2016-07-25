# coding:utf-8
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.data import astronaut
import matplotlib.pyplot as plt
import math
import copy
import scipy.stats as st


a = [[1,2,3],
     [4,5,6],
     [7,8,9]]

print np.delete(a,1, axis=0)




# img = cv2.imread('../image/rock/spiritsol118navcam.jpg',0)
# img = img[400:800,400:800]

cv2.waitKey(-1)