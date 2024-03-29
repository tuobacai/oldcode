# coding:utf-8
'''
基于lstm的孪生网络
step 1:网络结构
step 2:loss,opt,绑定
step 3:saver
'''
import tensorflow as tf
class SiameseLstm(object):

    '''双向lstm'''
    def bi_lstm(self,rnn_size, layer_size, keep_prob):
        #forward rnn
        with tf.name_scope('fw_rnn'),tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list=[tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_fw_cell_m=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),output_keep_prob=keep_prob)
        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=keep_prob)

        return lstm_fw_cell_m,lstm_bw_cell_m

    '''定义偏差变量'''
    def bias_variables(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    '''定义权重变量'''
    def weight_variables(self,shape,name):
        return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),name=name)

    '''转换输入'''
    def transform_inputs(self,inputs, rnn_size, sequence_length):
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, rnn_size])
        inputs = tf.split(inputs, sequence_length, 0)
        return inputs

    '''计算损失'''
    def contrastive_loss(self, Ew, y):
        l_1 = 0.25 * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_sum(y * l_1 + (1 - y) * l_0)
        return loss

    def __init__(self,config,is_training=True):
        self.input_x1=tf.placeholder(tf.int32, shape=[None, config.sequence_length], name='input_x')
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name='input_y')
        self.y_data = tf.placeholder(tf.float32, shape=[None], name='y_data')

        with tf.device('/cpu:0'):
            embedding=self.weight_variables([config.vocab_size,config.rnn_size],'embedding')
            inputs_x1 = tf.nn.embedding_lookup(embedding, self.input_x1)
            inputs_x2 = tf.nn.embedding_lookup(embedding, self.input_x2)

        inputs_x1 = self.transform_inputs(inputs_x1, config.rnn_size, config.sequence_length)
        inputs_x2 = self.transform_inputs(inputs_x2, config.rnn_size, config.sequence_length)

        with tf.variable_scope('output'):
            bilstm_fw, bilstm_bw = self.bi_lstm(config.rnn_size, config.layer_size, config.dropout_keep_prob)
            outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x1,
                                                                       dtype=tf.float32)
            output_x1 = tf.reduce_mean(outputs_x1, 0)
            ## 开启变量重用的开关
            tf.get_variable_scope().reuse_variables()

            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x2,
                                                                       dtype=tf.float32)
            output_x2 = tf.reduce_mean(outputs_x2, 0)

        with tf.variable_scope('dense_layer'):
            fc_w1 = self.weight_variables([2 * config.rnn_size, 128], 'fc_w1')
            fc_w2 = self.weight_variables([2 * config.rnn_size, 128], 'fc_w2')

            fc_b1 = self.bias_variables([128], 'fc_b1')
            fc_b2 = self.bias_variables([128], 'fc_b2')

            self.logits_1 = tf.matmul(output_x1, fc_w1) + fc_b1
            self.logits_2 = tf.matmul(output_x2, fc_w2) + fc_b2

        print 'fw(x1) shape: ', self.logits_1.shape
        print 'fw(x2) shape: ', self.logits_2.shape

        # calc Energy 1,2 ..
        f_x1x2 = tf.reduce_sum(tf.multiply(self.logits_1, self.logits_2), 1)
        norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_1), 1))
        norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_2), 1))
        self.Ew = f_x1x2 / (norm_fx1 * norm_fx2)

        print 'Ecos shape: ', self.Ew.shape

        # saver
        self.saver = tf.train.Saver(tf.global_variables())

        if not is_training:
            return

        # contrastive loss
        self.cost = self.contrastive_loss(self.Ew, self.y_data)

        # train optimization
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.grad_clip)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))







