#coding:utf-8
import numpy as np
import tensorflow as tf
import os
from datautils import DataUtils
from mlpmodel import MlpModel
from sklearn import metrics

#定义参数
class Config(object):
    learning_rate = 0.0005
    training_epochs = 500
    batch_size = 100
    n_hidden_1 = 300
    n_hidden_2 = 300
    n_class = 2
    display_step = 100

#mlp模型训练
class Train():
    def __init__(self):
        self.utils=DataUtils()
        self.mlp=MlpModel()
        self.config=Config()

    def run(self):
        #加载数据ß
        labels, features=self.utils.load_train_data(filename='/Users/yingjie10/deeptext/data/semanticmatch/data_200pos.txt')
        kDataTrain, kDataTrainC, kDataTest, kDataTestC = self.utils.kfold(features, labels)

        # 创建模型保存目录
        model_dir = "mlpmodel"
        model_name = "match"
        self.mkdir(model_dir)

        acc=[]
        for index in range(len(kDataTrain)):
            #处理数据
            print "cross validation:", index
            ty, tx = kDataTrainC[index], kDataTrain[index]
            testy, testx = kDataTestC[index], kDataTest[index]
            ty = self.utils.dense_to_one_hot(ty)
            testy = self.utils.dense_to_one_hot(testy)

           #特征维数
            n_input = tx.shape[1]

            # tf Graph input
            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, self.config.n_class])

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

            # Define loss and optimizer
            cost = tf.reduce_mean(-tf.reduce_sum(pred * tf.log(y), reduction_indices=[1]))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate).minimize(cost)

            # 定义模型保存对象
            saver = tf.train.Saver()

            # Initializing the variables
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                for epoch in range(self.config.training_epochs):
                    avg_cost = 0
                    total_batch = int(tx.shape[0] / self.config.batch_size)
                    for start,end in zip(range(0, len(tx), self.config.batch_size),range(self.config.batch_size, len(tx), self.config.batch_size)):
                        _, loss = sess.run([optimizer, cost], feed_dict={x: tx[start:end], y: ty[start:end]})
                        avg_cost += loss / total_batch

                    if epoch % self.config.display_step == 0:
                        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print ("Optimization Finished!")

                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
                print 'acc:',accuracy
                result = accuracy.eval({x: testx, y: testy})
                print("Accuracy:", result)
                acc.append(result)

                # 保存模型
                saver.save(sess, os.path.join(model_dir, model_name,str(index)))
                print("保存模型成功！")

        print "cross validation result"
        print "accuracy:", np.mean(acc)

        print("训练完成！")

    '''
    创建目录
    '''
    def  mkdir(self,model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)


if __name__=="__main__":
    mt=Train()
    mt.run()




