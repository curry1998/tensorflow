#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:44:44 2018

@author: bosh
"""
import tensorflow as tf
learn_rate=0.01
batch_size=16
epoch_step=10000
display_step=100#每100步打印数据

x=tf.placeholder(tf.float32,[None,20])#定义输入层含有20神经元
y=tf.placeholder(tf.float32,[None,5])#定义输出层含有5个神经元
#定义两个隐藏层的个数
layer1=16
layer2=32
#初始化定义参数
w={"h1":tf.Variable(tf.random_normal([20,layer1])),
   "h2":tf.Variable(tf.random_normal([layer1,layer2])),
   "out":tf.Variable(tf.random_normal([layer2,5]))
   }
b={"h1":tf.Variable(tf.random_normal([layer1])),
   "h2":tf.Variable(tf.random_normal([layer2])),
   "out":tf.Variable(tf.random_normal(5))
   }
def network(x_input,weight,biases):
    net1=tf.nn.relu(tf.matmul(x_input,weight["h1"])+biases["h1"])
    net2=tf.nn.relu(tf.matmul(net1,weight["h2"])+biases["h2"])
    out=tf.nn.relu(tf.matmul(net2,weight["out"])+biases["out"])
    return out
predict=network(x,w,b)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,y))
optimazer=tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
#预测正确的标签
correct_predic=tf.equl(tf.argmax(y,1),tf.argmax(predict,1))
accurate=tf.reduce_mean(tf.cast(correct_predic),tf.float32)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_step):
        avg_cost=0
        total_batch=int(alldata/batch_size)
        
        
        
        
        
        
        
        
        
        
        











