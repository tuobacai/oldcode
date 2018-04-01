#coding:utf-8
'''
这个类主要定义神经网络机构,包含四个部分:
step 1:定义神经网络forward时的计算
step 2:定义loss,选定优化器,并指定优化器优化loss
step 3:定义参数,变量,saver等
'''
import tensorflow as tf
class MlpModel(object):

    def __init__(self, config, is_training=True):
        # 定义参数
        # Store layers weight & biases
        weights = {
            # you can change
            'h1': tf.Variable(tf.random_normal([config.n_input, config.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([config.n_hidden_1, config.n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([config.n_hidden_2, config.n_hidden_3])),
            'out': tf.Variable(tf.random_normal([config.n_hidden_3, config.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([config.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([config.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([config.n_hidden_3])),
            'out': tf.Variable(tf.random_normal([config.n_classes]))
        }

        # 定义placeholder
        # tf Graph input
        self.input_data = tf.placeholder("float", [None, config.n_input])
        self.target = tf.placeholder("float", [None, config.n_classes])
        # self.input_data = tf.placeholder(tf.int32, [None, 784])
        # self.target = tf.placeholder(tf.float32, [None, 10])

        # step 1:定义forward计算
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, 0.75)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)

        out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
        self.pred = tf.nn.softmax(out_layer)

        # 如果是预测,那么不执行后面阶段
        if not is_training:
            return

        # step 2:定义loss,选定优化器,并指定优化器优化loss
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.target))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.cost)
        # self.cross_entropy = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred))
        # self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        self.saver = tf.train.Saver(tf.global_variables())
