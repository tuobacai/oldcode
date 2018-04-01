# coding:utf-8
'''
在tensorflow中的计算可以表示为一个有向图，或者称为计算图，其中每一个运算操作将作为一个节点，
节点与节点之间的连接称为边。这个计算图描述了数据的计算流程，它也负责维护和更新状态，
计算图中每一个节点可以有任意多个输入和任意多个输出，每一个节点描述了一种运算操作，
节点可以算是运算操作的实例化。
在计算图的边中流动(flow)的数据被称为张量(tensor),顾得名TensorFlow。
tensorflow运算过程可以分为两个阶段:
Phase 1: 构造图
Phase 2: 利用session来执行图中的操作。
'''
import tensorflow as tf
#第一阶段:构造计算图
a=5
b=3
c=tf.multiply(a,b)
d=tf.add(a,b)
e=tf.add(c,d)

#第二阶段:利用session来执行计算图中的操作
with tf.Session() as sess:
    print sess.run(e)



