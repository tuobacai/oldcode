# coding:utf-8
import tensorflow as tf
with tf.name_scope("hello") as name_scope:
    arr1 = tf.get_variable("arr1", shape=[2,10],dtype=tf.float32)

    print name_scope
    print arr1.name
    print "scope_name:%s " % tf.get_variable_scope().original_name_scope


