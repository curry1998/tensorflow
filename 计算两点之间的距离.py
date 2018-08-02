#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:27:34 2018

@author: bosh
"""

import tensorflow as tf
a=tf.constant([[1,2],[3,4],[9,5]],name="a")
b=tf.constant([[4,5],[7,8],[7,8]],name="b")
result=tf.add(a,b)
print(result)