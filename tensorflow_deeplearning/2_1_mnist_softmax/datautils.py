#coding:utf-8
'''
这个类主要是处理数据的操作
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
class DataUtils(object):

    '''
    加载数据
    '''
    def load_data(self,config):
        # Import data
        mnist = input_data.read_data_sets(config.data_dir, one_hot=True)
        return mnist

    '''
    得到批量数据
    '''
    def batch_iter(self, data, batch_size):
        # get_dataset_and_label
        x, y, mask_x = data
        x = np.array(x)
        y = np.array(y)
        data_size = len(x)
        num_batches_per_epoch = int((data_size - 1) / batch_size)
        for batch_index in range(num_batches_per_epoch):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_size)
            return_x = x[start_index:end_index]
            return_y = y[start_index:end_index]
            return_mask_x = mask_x[:, start_index:end_index]
            yield (return_x, return_y, return_mask_x)
