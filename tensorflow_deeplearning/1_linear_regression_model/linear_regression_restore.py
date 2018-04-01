# coding:utf-8
"""
线性模型y=w*x+b
tensorboard --logdir='/Users/yingjie10/PycharmProjects/tensorflow_introduction/linear_regression_model/my_graph/' --port 6006
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
#第一阶段:构造计算图

# Step 1: 从.xls中读数据
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: 为X创建占位符
X = tf.placeholder(tf.float32, name='X')

# Step 3: 创建权重变量w,偏移量b,初始化为0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: 建立预测模型
Y_predicted = X * w + b

#第二阶段:执行计算图中的计算
with tf.Session() as sess:
    # Step 5: 初始化所有变量
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "model_save/save_model.ckpt")

    # Step 6: 预测
    for x in data:
        predict=sess.run(Y_predicted,feed_dict={X:x})
        print 'predict {0}: {1}'.format(x[0], predict[0])






