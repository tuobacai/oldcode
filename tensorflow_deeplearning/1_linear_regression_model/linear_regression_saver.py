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

# Step 2: 为X和Y创建占位符
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: 创建权重变量w,偏移量b,初始化为0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: 建立预测模型
Y_predicted = X * w + b

# Step 5: 定义损失函数为平方损失
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: 学习算法为梯度下降算法,学习率为0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_model=optimizer.minimize(loss)

with tf.Session() as sess:
    # Step 7: 初始化所有变量
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./my_graph', sess.graph)

    # Step 8: 训练模型
    for i in range(100):  # train the model 100 times
        total_loss = 0
        for x, y in data:
            # feed and fetch
            _, l = sess.run([train_model, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print 'Epoch {0}: {1}'.format(i, total_loss / n_samples)
    save_path = saver.save(sess, "model_save/save_model.ckpt")

    # 关闭writer
    writer.close()

    # Step 9: 输出w,b
    w_value, b_value = sess.run([w, b])

# 画图
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
