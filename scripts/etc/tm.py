# -*- coding: utf-8 -*-
'''
    テンプレートマッチング
    Usage: $ tm.pys
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../../data/g-t_data/resized/spirit118-1.png',0)
img = img.astype(np.float32)
img2 = img.copy()
template = cv2.getGaborKernel(ksize = (11,11), sigma = 5,theta = np.pi, lambd = 11, gamma = 25/11,psi = np.pi * 1/2)
template = template.astype(np.float32)
print np.amax(template),np.amin(template)
# template = cv2.normalize(template, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
size = len(template)

w, h = template.shape[::-1]

# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            # 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
methods = ['cv2.TM_CCOEFF_NORMED']

for meth in methods: 
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)

    res = cv2.normalize(res, 0, 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
    img = img2.copy()
    mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    b,g,r = cv2.split(mask)
    print meth
    for th in range(180,125,-5):
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if res[i,j]>=th:
                    r[i+size/2,j+size/2]=255
        print th
        # r[res>th] = 255
        mask = cv2.merge((b,g,r))
        mask = mask.astype(np.uint8)
        cv2.imshow('{}'.format(meth),mask)
        cv2.waitKey(200)




    # print np.amax(res),np.amin(res)


cv2.waitKey(-1)




    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # cv2.rectangle(img,top_left, bottom_right, 255, 2)

    # 描画
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)

    # plt.show()