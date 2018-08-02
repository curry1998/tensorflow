#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:15:35 2018

@author: bosh
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("scrapy/MNIST_data",one_hot=True)
print(mnist.train.num_examples)