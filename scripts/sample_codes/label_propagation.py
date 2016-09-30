# -*- coding: utf-8 -*-
"""
    半教師あり学習(Label Propagation Algorithm)
    ラベルなしデータに，周囲にあるデータのラベルの数などから，新たにラベルをつける．
    実際に実装するときには，この後に改めてこのラベルを用いて教師あり学習(e.g. SVM)などで分類を行う．

    ==============================================
    Label Propagation learning a complex structure
    ==============================================
    Example of Label Propagation learning a complex internal structure
    to demonstrate "manifold learning". The outer circle should be
    labeled "red" and the inner circle "blue". Because both label groups
    lie inside their own distinct shape, we can see that the labels
    propagate correctly around the circle.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles

# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)

# label付け
outer, inner = 0, 1
labels = -np.ones(n_samples)
labels[0], labels[-1] = outer, inner

'''Learn with LabelSpreading'''
label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0) ## generate object
label_spread.fit(X, labels) ## fitting
output_labels = np.array(label_spread.transduction_) # 生成されたラベル

## Plot output labels
plt.figure(figsize=(8.5, 4))

## Origin  
plt.subplot(1, 2, 1)
plot_outer_labeled, = plt.plot(X[labels == outer, 0], X[labels == outer, 1], 'rs') ## それぞれのデータを色を分けてplot
plot_unlabeled,     = plt.plot(X[labels == -1, 0],    X[labels == -1, 1],    'g.')
plot_inner_labeled, = plt.plot(X[labels == inner, 0], X[labels == inner, 1], 'bs')

plt.legend((plot_outer_labeled, plot_inner_labeled, plot_unlabeled),
           ('Outer Labeled', 'Inner Labeled', 'Unlabeled'), loc='upper left',
           numpoints=1, shadow=False) ## データ名の表示
plt.title("Raw data (2 classes=red and blue)")

## Labeled
plt.subplot(1, 2, 2)
outer_numbers = np.where(output_labels == outer)[0]
inner_numbers = np.where(output_labels == inner)[0]
plot_outer, = plt.plot(X[outer_numbers, 0], X[outer_numbers, 1], 'rs')
plot_inner, = plt.plot(X[inner_numbers, 0], X[inner_numbers, 1], 'bs')
plt.legend((plot_outer, plot_inner), ('Outer Learned', 'Inner Learned'),
           loc='upper left', numpoints=1, shadow=False)
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
plt.show()
