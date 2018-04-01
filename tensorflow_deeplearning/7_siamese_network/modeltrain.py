# coding:utf-8
'''
孪生lstm模型训练
'''

import tensorflow as tf
from datautils import DataUtils
from siameselstm import SiameseLstm
import os
import time
import random

flags=tf.flags

tf.flags.DEFINE_string('train_file', 'siamese_pos187.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', '/data1/yingjie10/deeptext/data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'model', 'model save directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')

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


class ModelTrain(object):

    def __init__(self):
        self.config=Config()

    def train(self):

        print 'loading data...'
        train_data_loader=DataUtils(self.config,is_train=True)
        self.config.vocab_size=train_data_loader.vocab_size
        self.config.num_batches=train_data_loader.num_batches

        print 'begin train...'
        if self.config.init_from is not None:
            assert os.path.isdir(self.config.init_from), '{} must be a directory'.format(self.config.init_from)
            self.ckpt = tf.train.get_checkpoint_state(self.config.init_from)
            assert self.ckpt, 'No checkpoint found'
            assert self.ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        model=SiameseLstm(self.config,is_training=True)

        tf.summary.scalar('train_loss', model.cost)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            # restore model
            if  self.config.init_from is not None:
                model.saver.restore(sess, self.ckpt.model_checkpoint_path)

            for e in xrange(self.config.num_epochs):
                train_data_loader.reset_batch()
                b=0
                while not train_data_loader.eos:
                    b+=1
                    start=time.time()
                    x1_batch, x2_batch, y_batch = train_data_loader.next_batch()

                    # random exchange x1_batch and x2_batch
                    if random.random() > 0.5:
                        feed = {model.input_x1: x1_batch, model.input_x2: x2_batch, model.y_data: y_batch}
                    else:
                        feed = {model.input_x1: x2_batch, model.input_x2: x1_batch, model.y_data: y_batch}

                    fetches=[model.cost, merged, model.train_op]

                    train_loss,summary,_=sess.run(fetches=fetches,feed_dict=feed)
                    end = time.time()
                    print '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(e * self.config.num_batches + b,
                                                                                              self.config.num_epochs * self.config.num_batches,
                                                                                              e, train_loss,
                                                                                              end - start)
                    if (e *  self.config.num_batches + b) % 500 == 0:
                        checkpoint_path = os.path.join(self.config.save_dir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=e * self.config.num_batches + b)
                        print 'model saved to {}'.format(checkpoint_path)

                    if b % 20 == 0:
                        train_writer.add_summary(summary, e * self.config.num_batches + b)
if __name__=='__main__':
    modeltrain=ModelTrain()
    modeltrain.train()













