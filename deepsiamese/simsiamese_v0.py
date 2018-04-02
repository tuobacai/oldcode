# coding:utf-8
'''
相似度匹配网络
'''
import tensorflow as tf
from tensorflow.contrib import rnn
class SimSiamese(object):

    def __init__(self,config,is_training=True):
        self.config=config

        '''step 1:数据占位符'''
        self.input_x1 = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name='input_x')
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name='input_y')
        self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')

        '''step 2:网络结构'''
        #输入层
        self.inputs_x1, self.inputs_x2=self.handle_inputs()

        #encode层
        self.output_x1_pooled, self.output_x2_pooled=self.lstm_encoder()

        # 特征组合层
        self.mat_fea=self.feature_match()

        #mlp输出层
        self.logits =self.mlp_classifier()

        '''step 3:saver'''
        self.saver = tf.train.Saver(tf.global_variables())

        if not is_training:
            return

        '''step 4:loss'''
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits + 1e-10, self.y_data)
            self.cost = tf.reduce_mean(loss)

        '''step 5:opt'''
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.grad_clip)
        opt = tf.train.GradientDescentOptimizer(1e-3)
        self._train_op = opt.apply_gradients(zip(grads, tvars),
                                             global_step=tf.contrib.framework.get_or_create_global_step())

    '''输入层'''
    def handle_inputs(self):
        with tf.device('/cpu:0'),tf.variable_scope('embedding'):
            embedding = self.weight_variables([self.config.vocab_size, self.config.hidden_size], 'embedding')
            inputs_x1 = tf.nn.embedding_lookup(embedding, self.input_x1)
            inputs_x2 = tf.nn.embedding_lookup(embedding, self.input_x2)

        inputs_x1 = self.transform_inputs(inputs_x1, self.config.hidden_size, self.config.sequence_length)
        inputs_x2 = self.transform_inputs(inputs_x2, self.config.hidden_size, self.config.sequence_length)
        return inputs_x1,inputs_x2

    '''encoder部分'''
    def lstm_encoder(self):
        with tf.variable_scope('bilstm'):
            lstm_fw_cell, lstm_bw_cell = self.bi_lstm()
            outputs_x1,_, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, self.inputs_x1,dtype=tf.float32)
            ## 开启变量重用的开关
            tf.get_variable_scope().reuse_variables()
            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, self.inputs_x2,dtype=tf.float32)


        output_x1 = tf.concat(outputs_x1, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        output_x1_pooled = tf.reduce_max(output_x1,
                                         axis=1)  # [batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hid

        output_x2 = tf.concat(outputs_x2, axis=2)
        output_x2_pooled = tf.reduce_max(output_x2, axis=1)
        return output_x1_pooled,output_x2_pooled

    '''特征组合层'''
    def feature_match(self):
        with tf.variable_scope('feature match'):
            product_fea = tf.matmul(self.output_x1_pooled, self.output_x2_pooled)
            sub_fea = tf.abs(self.output_x1_pooled - self.output_x2_pooled)
            res_fea = tf.concat(1, [self.output_x1_pooled, self.output_x2_pooled, sub_fea, product_fea])
        return res_fea

    '''mlp输出层'''
    def mlp_classifier(self):
        with tf.variable_scope('mlp_layer'):
            input_w = tf.get_variable(name='input_w', shape=[self.config.hidden_size * 4, self.config.fc_dim],
                                      dtype=tf.float32)
            input_b = tf.get_variable("input_b", [self.config.fc_dim], dtype=tf.float32)
            layer_input = tf.add(tf.matmul(self.mat_fea, input_w), input_b)

            hidden_w = tf.get_variable(name='hidden_w', shape=[self.config.fc_dim, self.config.fc_dim],
                                       dtype=tf.float32)
            hidden_b = tf.get_variable("hidden_b", [self.config.fc_dim], dtype=tf.float32)
            layer_hidden = tf.add(tf.matmul(layer_input, hidden_w), hidden_b)

            output_w = tf.get_variable(name='output_w', shape=[self.config.fc_dim, self.config.num_classes],
                                       dtype=tf.float32)
            output_b = tf.get_variable("output_b", [self.config.num_classes], dtype=tf.float32)
            layer_output = tf.add(tf.matmul(layer_hidden, output_w), output_b)

        return layer_output

    '''双向lstm'''
    def bi_lstm(self):
        #forward rnn
        with tf.name_scope('fw_rnn'),tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list=[tf.contrib.rnn.LSTMCell(self.config.hidden_size) for _ in xrange(self.config.layer_size)]
            lstm_fw_cell_m=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),output_keep_prob=self.config.dropout_keep_prob)
        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.config.hidden_size) for _ in xrange(self.config.layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.config.dropout_keep_prob)

        return lstm_fw_cell_m,lstm_bw_cell_m

    '''定义偏差变量'''
    def bias_variables(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    '''定义权重变量'''
    def weight_variables(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    '''转换输入'''
    def transform_inputs(self, inputs, rnn_size, sequence_length):
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, rnn_size])
        inputs = tf.split(inputs, sequence_length, 0)
        return inputs
