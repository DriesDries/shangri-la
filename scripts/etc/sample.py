# -*- coding: utf-8 -*-
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier ,GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score


# 学習データを用意する
iris     = datasets.load_iris() #ライブラリ付属のサンプルデータ
features = iris.data            #特徴量の各データ 
                                #上記の分類の例でいうと、天気、場所、月日や各マンガを読んだかどうかに相当します
labels   = iris.target          #各データに対する正解ラベル
                                #特徴量に対する正解データ
                                #上記の分類の例でいうと、気温や性別に相当します

print features.shape,labels.shape


#特徴量の次元を圧縮
#似たような性質の特徴を同じものとして扱います
lsa = TruncatedSVD(2)
reduced_features = lsa.fit_transform(features)

print 'after',lsa,reduced_features.shape
print type(lsa)

#どのモデルがいいのかよくわからないから目があったやつとりあえずデフォルト設定で全員皆殺し 
clf_names = ["LinearSVC","AdaBoostClassifier","ExtraTreesClassifier" ,"GradientBoostingClassifier","RandomForestClassifier"]
for clf_name in clf_names:
  clf    = eval("%s()" % clf_name) #%sにclf_name入れてるだけ?clf_names()が入る
                                   #evalは、入力が文字列で、出力がパラメータのリストになる
                                   #clfが分類器になる
  scores = cross_val_score(clf,reduced_features, labels,cv=5)
  ''''sklearn.cross_validation.cross_val_score(分類器,特徴量のリスト,ラベルのリスト,cvはわからん)'''
  score  = sum(scores) / len(scores)  #モデルの正解率を計測
  print "%sのスコア:%s" % (clf_name,score)

# result
#LinearSVCのスコア:0.973333333333
#AdaBoostClassifierのスコア:0.973333333333
#ExtraTreesClassifierのスコア:0.973333333333
#GradientBoostingClassifierのスコア:0.966666666667
#RandomForestClassifierのスコア:0.933333333333