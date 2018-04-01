#coding:utf-8
'''
这个类主要调用模型,实现预测
'''
import tensorflow as tf
import os
from datautils import DataUtils
from softmodel import SoftModel

flags=tf.app.flags
FLAGS=flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", "./model", "model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
flags.DEFINE_string('test_path','','the absolute path of test file')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"softmax_outputs")),'output directory')
flags.DEFINE_float('lr',0.5,'the learning rate')
flags.DEFINE_integer('batch_size',100,'the batch_size of the training procedure')
flags.DEFINE_float('init_scale',0.1,'init scale')


class Config(object):
    train_path=FLAGS.train_path
    out_dir = FLAGS.out_dir
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size
    init_scale=FLAGS.init_scale
    model_dir=FLAGS.model_dir


class ModelPredict(object):

    def __init__(self):
        self.utils=DataUtils()
        self.config=Config()

    def predict(self):
        print("loading the dataset...")
        mnist = self.utils.load_data(config=self.config)

        print("begin predict...")
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-1 * self.config.init_scale, 1 * self.config.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                softmodel = SoftModel(config=self.config, is_training=False)
            ckpt=tf.train.get_checkpoint_state(self.config.model_dir)
            softmodel.saver.restore(session, ckpt.model_checkpoint_path)

            res=session.run(softmodel.pred,feed_dict={softmodel.input_data:mnist[0],softmodel.target:mnist[1]})

