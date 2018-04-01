# coding:utf-8
'''
session是客户使用tensorflow时的交互式接口，用户可以通过session生成计算图，
并通过session的run方法执行计算图。
客户端使用会话来和TF系统交互，
一般的模式是，建立会话，此时会生成一张空图；然后执行计算。
'''
import tensorflow as tf

#第一种形式
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
sess=tf.Session()
print sess.run(x)
sess.close()

#第二种形式
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
with tf.Session() as sess:
    sess.run(x)
