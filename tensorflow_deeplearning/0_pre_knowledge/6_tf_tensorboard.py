# coding:utf-8
'''
tensorboard简单使用介绍
step 1:安装tensorboard
step 2:tensorboard --logdir='/Users/yingjie10/PycharmProjects/tensorflow_introduction/pre_knowledge/graphs'  --port 6006
step 3:在浏览器输入地址,eg.http://192.168.2.1:6006
'''
import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
with tf.Session() as sess:
    writer=tf.summary.FileWriter('graphs',sess.graph)
    print sess.run(x)
    writer.close()

