#-*-coding:utf-8-*-
"""
    オンライン学習のひとつであるPassive-Aggresive Algorithmの実装．

    時間軸でより新しいデータの重みを大きくして学習させる．


    ・過去のデータはいらないのか？

"""

import numpy as np

class PassiveAggressive:
  
    def __init__(self, feat_dim):
        self.t = 0
        self.w = np.ones(feat_dim)

    def _get_eta(self, l, feats):
        return l/np.dot(feats, feats)

    def train(self, y_vec, feats_vec):
        for i in range(len(y_vec)):
            self.update(y_vec[i], feats_vec[i,])

    def predict(self, feats):
        return np.dot(self.w, feats)

    def update(self, y, feats):
        l = max([0, 1-y*np.dot(self.w, feats)])
        eta = self._get_eta(l, feats)
        self.w += eta*y*feats
        self.t += 1
        return 1 if l == 0 else 0


class PassiveAggressive1(PassiveAggressive):
    def __init__(self, feat_dim, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self, feat_dim)

    def _get_eta(self, l, feats):
        return min(self.c, l/np.dot(feats, feats))


class PassiveAggressive2(PassiveAggressive):
    def __init__(self, feat_dim, c=0.1):
        self.c = c
        PassiveAggressive.__init__(self, feat_dim)

    def _get_eta(self, l, feats):
        return l/(np.dot(feats, feats)+1/(2*self.c))


if __name__ == '__main__':
    