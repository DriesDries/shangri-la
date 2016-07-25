# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as rd # 乱数に関するクラス
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import pandas


''' 乱数生成 '''
size = 10000
rate = 0.01

# 自分で指定した生起分布に基づいた乱数
ori_rn = np.sort(rd.choice(a=5, size=size, p=[0.1,0.3,0.3,0.2,0.1]))

# これにも生起分布をもたせたい
_ = rd.choice(a=size, size=size*rate, replace=False)

sel_rn = []
for i in _:
    sel_rn = np.append(sel_rn,ori_rn[i])
# sel_rn = sel_rn.astype(np.uint16)


ori_rn = ori_rn.astype(np.float64)

''' display result '''
plt.figure(figsize=(14,6)) # グラフの大きさの指定、縦x横[inch]

# 自分で指定した生起分布に基づく乱数配列
plt.subplot(131)
plt.title('original_%s'%int(size))
y, bins, patches = plt.hist(ori_rn, bins=5, normed=1, facecolor='green', alpha=0.5)
bins = np.delete(bins,5,0)
plt.plot(bins,y,color='blue')

# rnからランダムで取り出した配列
plt.subplot(132)
plt.title('select_%s'%(int(size*rate))+'/%s'%int(size))
y, bins, patches = plt.hist(sel_rn, bins=5, normed=1, facecolor='green', alpha=0.5)
bins = np.delete(bins,5,0)
plt.plot(bins,y,color='blue')

# 一様分布の乱数配列
plt.subplot(133)
plt.title('random_%s'%(int(size*rate))+'/%s'%int(size))
y, bins, patches = plt.hist(_, bins=5, normed=1, facecolor='green', alpha=0.5)
bins = np.delete(bins,5,0)
plt.plot(bins,y,color='blue')
# p_ = pandas.Series(_)
# print p_.describe()


plt.show()












# 自分で指定したの
# rn = [2, 2, 4, 6, 4, 5, 2, 3, 1, 2, 0, 4, 3, 3, 3, 3, 4, 2, 7, 2, 4, 3, 3, 3, 4, 3, 7, 5, 3, 1, 7, 6, 4, 6, 5, 2, 4, 7, 2, 2, 6, 2, 4, 5, 4, 5, 1, 3, 2, 3]

# 指定した次元数の乱数
# rn = rd.rand(100,) # 次元と要素数を指定できる、毎回異なる値
# rn = rd.random_sample((10,)) # 指定した要素数の乱数生成
# rn = sigma * rd.randn(100,) +mu # 正規分布に基づいた乱数、sigmaとmuは標準偏差と平均


# rn = rd.choice(a=15, size=10, replace=False) # range(a)の範囲からsizeで指定された要素数の乱数を生成
                                           # replaceがあると、同じ値をとることがある
# _ = rd.randint(low=0, high=5, size=size) # lowからhigh-1の範囲でsizeの要素数の離散一様分布
# plt.figure(figsize=(横, 縦), dpi=解像度, facecolor=背景色)