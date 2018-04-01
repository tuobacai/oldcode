#coding:utf-8
'''
这个类主要定义神经网络机构,包含四个部分:
step 1:定义神经网络forward时的计算
step 2:定义loss,选定优化器,并指定优化器优化loss
step 3:迭代的对数据进行训练
step 4:准确率评测
'''
import tensorflow as tf

class SoftModel(object):

    '''
    这部分一般完成step 1和step 2
    '''
    def __init__(self,config,is_training=True):
        #定义参数
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        #定义placeholder
        self.input_data = tf.placeholder(tf.int32, [None, 784])
        self.target = tf.placeholder(tf.float32, [None, 10])

        #step 1:定义forward计算
        self.pred=tf.nn.softmax(tf.matmul(self.input_data, W) + b)

        #如果是预测,那么不执行后面阶段
        if not is_training:
            return

        #step 2:定义loss,选定优化器,并指定优化器优化loss
        self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred))
        self.train_step=tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        self.saver=tf.train.Saver(tf.global_variables())

    '''
    这部分完成step 3
    '''
    def train_step(self,mnist,sess,train_step):

        #step 3:迭代的对数据进行训练
        sess=tf.InteractiveSession()
        init=tf.global_variables_initializer()
        for i in range(1000):
            batch_xs,batch_ys=mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={self.input_data:batch_xs,self.target:batch_ys})


    '''
    这部分完成step 4
    '''
    def evaluate(self,sess,pred,target,mnist,model):
        correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(target,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={model.input_data: mnist.test.images,
                                            model.target: mnist.test.labels}))







