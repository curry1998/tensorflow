#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:22:20 2018

@author: bosh
"""

#placeholder占位符从外界传入一个值可以接受任意维度的tensor（及传入参数）
import tensorflow as tf
#import numpy as np

a=tf.placeholder(tf.float32)#声明a作为placeholder接收float32的数据
b=tf.placeholder(tf.float32)#声明b作为placeholder接收float32的数据
adder_node=a+b#对两个placeholder输入的tensor数据做加法
c=tf.placeholder(tf.float32)#声明c作为placeholder接收float32的数据
adder_and_triple= adder_node*c #执行（a+b)*c

with tf.Session() as sess:
    print(sess.run(adder_node,{a:5,b:7}))#这里placeholder输入的为rank=0的tensor数据
    print(sess.run(adder_node,{a:[[6,9],[6,10],[12,3]],b:[[4,5],[36,13],[23,35]]}))#placeholder输入的位rank=2的tensor数据
    print(sess.run(adder_and_triple,{a:[8,9],b:[1,6],c:3}))#a，b为rank=1的tensor数