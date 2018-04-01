#coding:utf-8
'''
数据处理
'''
import numpy as np
from sklearn.cross_validation import KFold
class DataUtils(object):
    '''
    加载训练数据
    '''
    def load_train_data(self, filename):
        labels = []
        feature = []
        i = 0
        pos = 0
        neg = 0
        for line in open(filename):
            if i == 0:
                temp = line.strip().split(',')
                feaname = temp[3:]
                print 'feaname:', feaname
            else:
                temp = line.strip().split('`')
                label = int(temp[0])
                # 只取用几个简单的特征
                fea = [temp[i] for i in range(len(temp)) if i > 2 and i != 4 and i < 9]
                interfea = temp[4]
                itemp = interfea.split('~')
                # 正样本取100个,负样本取200个
                if label == 0:
                    neg += 1
                if label == 1:
                    pos += 1
                if label == 0 and neg < 200:
                    labels.append(label)
                    feature.append(fea)
                if label == 1 and pos < 100:
                    labels.append(label)
                    feature.append(fea)
            i += 1
        return labels, feature

    '''
    加载数据
    '''
    def load_predict_data(self, filename):
        labels = []
        feature = []
        i = 0
        pos = 0
        neg = 0
        for line in open(filename):
            if i == 0:
                temp = line.strip().split(',')
                feaname = temp[3:]
                print 'feaname:', feaname
            else:
                temp = line.strip().split('`')
                label = int(temp[0])
                # 只取用几个简单的特征
                fea = [temp[i] for i in range(len(temp)) if i > 2 and i != 4 and i < 9]
                interfea = temp[4]
                itemp = interfea.split('~')
                # 正样本取100个,负样本取200个
                if label == 0:
                    neg += 1
                if label == 1:
                    pos += 1
                if label == 0 and neg < 20:
                    labels.append(label)
                    feature.append(fea)
                if label == 1 and pos < 10:
                    labels.append(label)
                    feature.append(fea)
            i += 1
        return labels, feature

    '''
    处理多分类的类别
    '''
    def dense_to_one_hot(self, labels_dense, num_classes=2):
        """ convert class lables from scalars to one-hot vector"""
        labels_dense = np.asarray(labels_dense)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    '''
    k层交叉验证
    '''
    def kfold(self, trainData, trainClass, nFold=10):
        skf = KFold(len(trainData), nFold, shuffle=True, random_state=1234)
        kDataTrain = []
        kDataTrainC = []
        kDataTest = []
        kDataTestC = []

        trainData = np.asarray(trainData)
        trainClass = np.asarray(trainClass)
        for train_index, test_index in skf:
            X_train, X_test = trainData[train_index], trainData[test_index]
            y_train, y_test = trainClass[train_index], trainClass[test_index]
            kDataTrain.append(X_train)
            kDataTrainC.append(y_train)
            kDataTest.append(X_test)
            kDataTestC.append(y_test)
        return kDataTrain, kDataTrainC, kDataTest, kDataTestC