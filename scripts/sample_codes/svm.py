# -*- coding: utf-8 -*-
'''

Usage: $ python template.py <argv>
''' 

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC


class SupportVectorMachine():

    def linearSVM(self):
        '''
            線形SVMを用いた2クラス分類
            args :      -> 
            dst  :      -> 
            param:      -> 
        '''
        # 学習データ
        data_training_tmp = np.loadtxt('../../../data/statistical_data/CodeIQ_auth.txt', delimiter=' ')
        data_training = [[x[0], x[1]] for x in data_training_tmp]
        label_training = [int(x[2]) for x in data_training_tmp]

        # 試験データ
        data_test = np.loadtxt('../../../data/statistical_data/CodeIQ_auth.txt', delimiter=' ')

        print np.array(data_test).shape

        # 学習
        estimator = LinearSVC(C=1.0)
        estimator.fit(data_training, label_training)

        # 予測
        label_prediction = estimator.predict(data_test[:,0:2])
        print(label_prediction)

        print 




if __name__ == '__main__':

    svm = SupportVectorMachine()

    svm.main()