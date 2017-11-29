#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:13:13 2017

@author: moonsooyoung
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
#mglearn.plots.plot_knn_classification(n_neighbors=1)

from sklearn.model_selection import train_test_split

from random import * 

def createDataSet():
    mu, sigma = 0.0, 1.0
    X_train=np.random.normal(mu, sigma, size=(1,20))
    X_train1=X_train+1
    X_train2=X_train1+1
    Xtrain=np.append(X_train, X_train1 ,axis=0)
    Xtrain1=np.append(Xtrain, X_train2,axis=0)
    Ytrain1=np.array([[0],[1],[2]])
    return Xtrain1, Ytrain1

#clf= KNeighborsClassifier(n_neighbors=9)


Xtrain1, Ytrain1 =  createDataSet()
data=knn.fit(Xtrain1,Ytrain1)  #학습시킴

def createTestSet():
    Xtest=np.array([[-0.09,0.0,0.5,0.2,-0.8,-0.6,-0.3,0.7,0.4,0.6,0.7,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,-0.9],
                    [1.5,1.0,1.5,1.2,-1.8,-1.6,-1.3,1.7,1.4,1.6,1.7,1.2,1.1,1.2,1.1,1.2,1.1,1.2,1.1,-1.9]])
        
    Ytest=knn.predict(Xtest)
    Ytest=np.array([[0.],
                    [1.]])
    return Xtest,Ytest

Xtest, Ytest = createTestSet()

print('테스트 예측: {}'.format(knn.predict(Xtest)))
print ('\n테스트 세트 정확도:\n',format(knn.score(Xtest, Ytest)))