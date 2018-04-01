# coding:utf-8
'''
常数
'''
import tensorflow as tf
import numpy as np

#常量张量
sess = tf.Session()
a = tf.constant(2, name="a")
data = tf.zeros(shape=[2, 3], dtype=tf.int32, name="input_data")
d_1 = tf.zeros_like(data)
d_2 = tf.zeros_like(data, tf.float32)
data1 = tf.zeros(shape = [2, 3], dtype = tf.int32, name = "input_data")
d_3 = tf.ones_like(data1)
d_4 = tf.ones_like(data1, tf.float32)
data2 = tf.fill([2,3], 9)
print sess.run(d_1)
print sess.run(d_2)
sess.close()

#序列操作
sess = tf.Session()
data = tf.linspace(10.0, 15.0, 10)
data1 = tf.range(3, 15, 3)
print sess.run(data1)
sess.close()

#随机数张量
data = tf.random_normal([2, 3])#这个函数返回一个随机数序列，数组里面的值按照正态分布
sess = tf.Session()
writer = tf.summary.FileWriter('graphs', sess.graph)
data1 = tf.random_uniform([2, 3])#这个函数返回一个随机数序列，数组里面的值按照均匀分布,数据范围是 [minval, maxval)
print sess.run(data)
writer.close()
sess.close()