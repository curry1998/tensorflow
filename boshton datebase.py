#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Jan 21 05:56:03 2018

@author: bosh
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
boshton=load_boston()
print(boshton.keys())
print(boshton.feature_names)
x=boshton.data[:,np.newaxis,5]#对data所有行的第六列,需要列表里面的元素还是列表则需要增加一个维度
y=boshton.target
lm=LinearRegression()
lm.fit(x,y)
print("拟合的曲线是y=",lm.coef_,"*x+",lm.intercept_)
print("拟合的比值为：",lm.score(x,y))
plt.scatter(x,y)
plt.plot(x,lm.predict(x),color="red")