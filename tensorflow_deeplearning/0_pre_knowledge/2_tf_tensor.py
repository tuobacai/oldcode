# coding:utf-8
'''
张量(tensor)，即任意维度的数据，一维、二维、三维、四维等数据统称为张量。
0-d tensor: scalar (number)
1-d tensor: vector
2-d tensor: matrix
'''
import tensorflow as tf
sess=tf.InteractiveSession()

#0-d tensor, or "scalar"
t_0 = 19
tf.zeros_like(t_0) # ==> 0
print "a:"
a=tf.ones_like(t_0) # ==> 1
print a.eval()
sess.close()


#1-d tensor, or "vector"
t_1 = [1,2,3]
tf.zeros_like(t_1) # ==> ['' '' '']
print "b:", tf.ones_like(t_1) # ==> TypeError: Expected string, got 1 of type 'int' instead.

#2x2 tensor, or "matrix"
t_2 = [
        [1,2],
        [3,4],
        [5,6]
       ]
tf.zeros_like(t_2) # ==> 2x2 tensor, all elements are False
c=tf.ones_like(t_2) # ==> 2x2 tensor, all elements are True
with tf.Session() as sess:
    print sess.run(c)

