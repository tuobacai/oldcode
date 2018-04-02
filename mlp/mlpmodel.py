#coding:utf-8
'''
定义Mlp网络结构
'''
import tensorflow as tf
class MlpModel(object):

    '''
    创建mlp网络结构
    '''
    def create_model(self, x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.relu(layer_2)

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

        return out_layer