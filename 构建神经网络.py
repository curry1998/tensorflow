#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:01:56 2018

@author: bosh
"""

import tensorflow as tf
import numpy as np
#添加层
def add_layer(inputs,input_size,output_size,active_function=None):
    weight=tf.Variable(tf.random_normal([input_size,output_size]))
    basis=tf.Variable(tf.zeros([1,output_size])+0.1)#basis为1行，output_size列的全部为0.1的矩阵
    plus=tf.matmul(inputs,weight)+basis#矩阵的乘法再加上basis
    if active_function==None:
        output=plus
    else:
        output=active_function(plus)
    return output
#产生数据
x_date=np.linspace(-1,1,200)[:,np.newaxis]
#增加数据的维度200行二维矩阵
#让其更像一个真实数据
noise=np.random.normal(0,0.05,x_date.shape)
y_data=np.square(x_date)-0.5
#y=x^2-0.5
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#隐藏层（输入层为一）
la=add_layer(xs,1,10,tf.nn.relu)
prediction=add_layer(la,10,1,None)
#loss函数
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction)))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,feed_dict={xs:x_date,ys:y_data})
    if i%50==0:
        print(sess.run(loss,{xs:x_date,ys:y_data}))

















    