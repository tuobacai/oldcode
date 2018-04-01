# coding:utf-8
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

#加载数据
x_data = np.loadtxt("data/ex4x.dat").astype(np.float32)
y_data = np.loadtxt("data/ex4y.dat").astype(np.float32)
scaler = preprocessing.StandardScaler().fit(x_data)
x_data_standard = scaler.transform(x_data)

#创建权重变量x和偏移量y
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))

#模型
y= 1 / (1 + tf.exp(-tf.matmul(x_data_standard, W) + b))
#损失
loss=tf.reduce_mean(- y_data.reshape(-1, 1) *  tf.log(y) - (1 - y_data.reshape(-1, 1)) * tf.log(1 - y))
#优化方法
optimizer = tf.train.GradientDescentOptimizer(1.3)
#训练
train = optimizer.minimize(loss)
#初始化所有变量
init = tf.initialize_all_variables()

#执行运算
sess = tf.Session()
sess.run(init)
for step in range(100):
    sess.run(train)
    if step % 10 == 0:
        print step, sess.run(W).flatten(), sess.run(b).flatten()
sess.close()