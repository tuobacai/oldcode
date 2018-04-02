#coding:utf-8
import numpy as np
import tensorflow as tf
import os
from datautils import DataUtils
from mlpmodel import MlpModel
from sklearn import metrics

#定义参数
class Config(object):
    n_hidden_1 = 300
    n_hidden_2 = 300
    n_class = 2


#mlp模型训练
class Predict():
    def __init__(self):
        self.utils=DataUtils()
        self.mlp=MlpModel()
        self.config=Config()

    def run(self):
        #加载数据
        label, feature=self.utils.load_predict_data(filename='/Users/yingjie10/deeptext/data/semanticmatch/data_200pos.txt')
        labels, features = np.asarray(label), np.asarray(feature)

        # 创建模型保存目录
        model_dir = "mlpmodel"
        model_name = "match"
        index=3

       #特征维数
        n_input = features.shape[1]

        # tf Graph input
        x = tf.placeholder("float", [None, n_input])

        # Store layers weight & biases
        weights = {

            'h1': tf.Variable(tf.random_normal([n_input, self.config.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.config.n_hidden_1, self.config.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.config.n_hidden_2, self.config.n_class]))
        }

        biases = {

            'b1': tf.Variable(tf.random_normal([self.config.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.config.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.config.n_class]))

        }

        # Construct model
        pred = self.mlp.create_model(x, weights, biases)

        # 定义模型保存对象
        saver = tf.train.Saver()

        # 预测函数
        y = tf.nn.softmax(pred)

        with tf.Session() as sess:
            # 恢复模型
            saver.restore(sess,os.path.join(model_dir, model_name,str(index)))

            # 预测
            result = sess.run(y, feed_dict={x: features})
            results=[res.argmax() for res in result]
            print 'result:', results
            auc=metrics.roc_auc_score(label,results)
            print 'auc:',auc


if __name__=="__main__":
    mp=Predict()
    mp.run()




