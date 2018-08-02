#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 02:54:56 2018

@author: bosh
"""
from sklearn.linear_model import LinearRegression
x=[[1,1],[2,2],[3,3]]
y=[1,2,3]
lm=LinearRegression()
lm.fit(x,y)
print(lm.predict([[4, 4]]))
print(lm.coef_,lm.intercept_)  
print(lm.score(x,y)) 