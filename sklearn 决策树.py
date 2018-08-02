#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:17:44 2018

@author: bosh
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
iris=load_iris()

dtc=DecisionTreeClassifier(random_state=1)
#dtr=DecisionTreeRegression()
#print(dtc.fit(iris.data,iris.target))
i=cross_val_score(dtc,iris.data,iris.target,cv=10) 
print(i)
