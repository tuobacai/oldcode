# coding:utf-8
'''
孪生lstm模型预测
针对训练数据进行预测,预测结果和实际label都是一样的,auc为1.0.
'''

import tensorflow as tf
from datautils import DataUtils
from siameselstm import SiameseLstm
import os
import time
import random
from sklearn import metrics

flags=tf.flags

tf.flags.DEFINE_string('train_file', 'siamese_pos187.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'siamese_pos187.txt', 'train raw file')
# tf.flags.DEFINE_string('data_dir', '/data1/yingjie10/deeptext/data', 'data directory')
tf.flags.DEFINE_string('data_dir', '/Users/yingjie10/deeptext/data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'model', 'model save directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('init_from',  'model', 'continue training from saved model at this path')

tf.flags.DEFINE_integer('rnn_size', 64,
                        'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 4, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')

FLAGS=flags.FLAGS

class Config(object):
    train_file=FLAGS.train_file
    test_file = FLAGS.test_file
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    log_dir = FLAGS.log_dir
    init_from = FLAGS.init_from
    rnn_size = FLAGS.rnn_size
    dropout_keep_prob = FLAGS.dropout_keep_prob
    layer_size = FLAGS.layer_size
    batch_size = FLAGS.batch_size
    sequence_length = FLAGS.sequence_length
    grad_clip = FLAGS.grad_clip
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate
    decay_rate = FLAGS.decay_rate
    vocab_size=0
    num_batches=0


class ModelPredict(object):

    def __init__(self):
        self.config=Config()

    def predict(self):

        print 'loading data...'
        test_data_loader=DataUtils(self.config,is_train=False)
        self.config.vocab_size=test_data_loader.vocab_size

        print 'creating model...'
        model = SiameseLstm(self.config, is_training=False)

        print 'begin train...'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # restore model
            print 'restore model...'
            model_checkpoint_path = os.path.join(self.config.save_dir, 'model.ckpt-500')
            model.saver.restore(sess,model_checkpoint_path)

            print 'predict...'
            feed = {model.input_x1: test_data_loader.x1, model.input_x2: test_data_loader.x2, model.y_data: test_data_loader.y}
            fetches=[model.logits_1,model.logits_2,model.Ew]
            result=sess.run(fetches=fetches,feed_dict=feed)
            results=[0 if i<0.5 else 1 for i in result[2]]
            auc = metrics.roc_auc_score(test_data_loader.y, results)
            print 'auc:', auc

if __name__=='__main__':
    modeltrain=ModelPredict()
    modeltrain.predict()













