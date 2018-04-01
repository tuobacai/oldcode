# coding:utf-8
'''
0、TF计算都是通过数据流图（Graph）来展现的，
一个数据流图包含一系列节点（OP）操作以及在节点之间流动的数据，
这些节点和数据流分别称之为计算单元和Tensor对象。
1、当进入TF时（例如import tensorflow as tf），TF内部会注册一个默认的Graph，
可通过 tf.get_default_graph()  来获得这个默认的Default_Graph，
只要简单地调用某个函数，就可以向这个默认的图里面添加操作（节点）。
'''
import tensorflow as tf

#step 1:default graph
g=tf.get_default_graph()
with g.as_default():
    a=3
    b=5
    c=tf.add(a,b)
sess=tf.Session(graph=g)
print sess.run(c)
sess.close()

#step 2:自己构造一个图
g=tf.Graph()
with g.as_default():
    multi_operate=tf.multiply(4,5)
sess=tf.Session(graph=g)
res=sess.run(multi_operate)
print res
sess.close()

#step 3:多个图
g1=tf.Graph()
g2=tf.Graph()
with g1.as_default():
    a = 3
    b=5
    c=tf.add(a,b)

with g2.as_default():
    a = 3
    b=5
    d=tf.add(a,b)
sess1=tf.Session(graph=g1)
sess2=tf.Session(graph=g2)
res1=sess1.run(c)
res2=sess2.run(d)
print res1,res2
sess1.close()
sess2.close()


'''
freestyle
'''
# g1 = tf.Graph()
# with g1.as_default():
#     c1 = tf.constant([1.0])
# with tf.Graph().as_default() as g2:
#     c2 = tf.constant([2.0])
#
# with tf.Session(graph=g1) as sess1:
#     print sess1.run(c1)
# with tf.Session(graph=g2) as sess2:
#     print sess2.run(c2)

# result:
# [ 1.0 ]
# [ 2.0 ]


# g = tf.Graph()
# with g.as_default():
#     x = tf.add(3, 5)
# # sess = tf.Session(graph=g)
# with tf.Session(graph=g) as sess:
#     print sess.run(x)
# sess.close()

# g = tf.Graph()
# with g.as_default():
#     a = 3
#     b = 5
#     x = tf.add(a, b)
# sess = tf.Session(graph=g) # session is run on the graph g
# print sess.run(x)
# sess.close()

# g1 = tf.get_default_graph()
# g2 = tf.Graph()
# # add ops to the default graph
# with g1.as_default():
#  a = tf.constant(3)
# # add ops to the user created graph
# with g2.as_default():
#  b = tf.constant(5)
# c = tf.constant(4.0)
# assert c.graph is tf.get_default_graph()




