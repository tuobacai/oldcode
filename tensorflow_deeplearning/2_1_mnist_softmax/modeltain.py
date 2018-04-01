#coding:utf-8
'''
这部分主要是模型训练部分
'''
import tensorflow as tf
import os
from datautils import DataUtils
from softmodel import SoftModel

flags=tf.app.flags
FLAGS=flags.FLAGS

flags.DEFINE_string('train_path','','the absolute path of train file')
flags.DEFINE_string('out_dir',os.path.abspath(os.path.join(os.path.curdir,"softmax_outputs")),'output directory')
flags.DEFINE_float('lr',0.5,'the learning rate')
flags.DEFINE_integer('batch_size',100,'the batch_size of the training procedure')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('num_epoch',60,'num epoch')
flags.DEFINE_integer('checkpoint_steps',10,'checkpoint steps every num epoch ')

class Config(object):
    train_path=FLAGS.train_path
    out_dir = FLAGS.out_dir
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size
    init_scale=FLAGS.init_scale
    num_epoch=FLAGS.num_epoch
    checkpoint_steps=FLAGS.checkpoint_steps

class ModelTrain(object):

    def __init__(self):
        self.utils=DataUtils()
        self.config=Config()

    def train_step(self):
        print("loading the dataset...")
        mnist=self.utils.load_data(config=self.config)

        print("begin training")
        with tf.Graph().as_default(), tf.Session() as session:
            initializer = tf.random_uniform_initializer(-1 * self.config.init_scale, 1 *  self.config.init_scale)
            with tf.variable_scope('model',reuse=None,initializer=initializer):
                softmodel = SoftModel(config=self.config, is_training=True)

            #add summary
            train_summary_dir=os.pardir.join(self.config.out_dir,'summaries','train')
            train_summary_writer=tf.summary.FileWriter(train_summary_dir, session.graph)

            #add checkpoint
            checkpoint_dir=os.path.abspath(os.path.join(self.config.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)


            tf.initialize_all_variables().run()
            global_steps = 1

            for i in range(self.config.num_epoch):
                print("the %d epoch training..." % (i + 1))
                global_steps=self.run_epoch()

                if i%self.config.checkpoint_steps==0:
                    path = softmodel.saver.save(session, checkpoint_prefix, global_steps)
                    print("Saved model chechpoint to{}\n".format(path))

            test_accuracy = self.evaluate(test_model, session, test_data)
            print("the test data accuracy is %f" % test_accuracy)
            print("program end!")


    def run_epoch(self):
        global_steps=1
        return global_steps

    def evaluate(self,test_model, session, test_data):
        acc=float(0)
        return acc
















