#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:47:20 2018

@author: bosh
"""

import tensorflow as tf
import numpy as np
#创建训练集
x_data=np.random.rand(200).astype(np.float32)
y_data=x_data*0.3+4
#创建tensorflow结构
weight=tf.Variable(tf.random_uniform([1],0,1))#创建一个0到1的随机数
bais=tf.Variable(tf.zeros([1]))#初始值设为1

y=weight*x_data+bais

#loss函数
loss=tf.reduce_mean(tf.square(y-y_data))
#loss函数最小化(学习的效率设为0.25)
#优化器选择梯度下降
optimizer=tf.train.GradientDescentOptimizer(0.25)
train=optimizer.minimize(loss)
#初始化变量
init=tf.initialize_all_variables()
#session激活
sess=tf.Session()
sess.run(init)#激活init
#进行训练
for step in range(501):
    sess.run(train)
    if step%50==0:
        print(step,sess.run(weight),sess.run(bais))











