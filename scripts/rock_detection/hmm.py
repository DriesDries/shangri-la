# -*- coding: utf-8 -*-
'''
    Hiden Markov Model
'''


import numpy as np
from hmmlearn import hmm

# GaussianHMM クラスから出力確率が正規分布に従う隠れマルコフモデルを作る。
# n_components 状態数
# covariance_type 共分散の種類
# modelを作成した後にも、startprobとtransmatとmeansとcovarsが必要となる
model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# 初期状態確率 π を指定する。
model.startprob_ = np.array([1., 0., 0.])

# 遷移確率 A を指定する。
# model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            # [0.3, 0.5, 0.2],
                            # [0.3, 0.3, 0.4]])

model.transmat_ = np.array([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 0., 0.]])


# 出力確率 B を指定する。
# ただし出力は正規分布に従うと仮定しているため、正規分布のパラメータの
# 平均値 μ (means_) と、共分散 σ^2 (covars_) を指定する。
model.means_ = np.array([[0.0  , 0.0  ],
                         [10.0 , 10.0 ],
                         [100.0, 100.0]])
model.covars_ = 2 * np.tile(np.identity(2), (3, 1, 1))

# sample信号の作成
X, Z = model.sample(10000)

a = 0
b = 0
c = 0

for z in Z:
    if z == 0:
        a = a + 1
    if z == 1:
        b = b + 1
    if z == 2:
        c = c + 1

print a,b,c



# 対数尤度の表示
# print model.score(X)
# print np.exp(model.score(X))
# print model.predict(X)

# 未知モデルからの推定
# remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
# model.fit(X)
# remodel.fit(X)