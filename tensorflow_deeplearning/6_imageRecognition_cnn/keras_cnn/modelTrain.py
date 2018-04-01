# coding:utf-8
import keras
import numpy as np
np.random.seed(123)  # reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# 期望的图片尺寸
img_width, img_height = 150, 150
train_data_dir = '/Users/yingjie10/Documents/data/train'
validation_data_dir = '/Users/yingjie10/Documents/data/validation'
# 训练集图片数量
train_samples = 2000
# 验证集图片数量
validation_samples = 800
# 运行轮数
epoch = 1
'''
我们的数据量很少,模型训练容易出现过拟合,在这里通过一系列随机变换对数据进行提升，
这样我们的模型将看不到任何两张完全相同的图片,
这有利于我们抑制过拟合，使得模型的泛化能力更好。
'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,dim_ordering='th')

'''
对测试集进行数据提升,只改变尺度
'''
test_datagen = ImageDataGenerator(rescale=1./255,dim_ordering='th')

#从文件中获取图像,进行数据提升
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

# ** 搭建网络 **
model = Sequential()
#输入为(3,150,150)的图像矩阵,
# 第一层卷积层，有32个卷积核（过滤器），每个卷积核的尺寸是3x3
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height),dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

#第二层卷积层，有32个卷积核（过滤器），每个卷积核的尺寸是3x3
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
#第三层卷积层，有64个卷积核（过滤器），每个卷积核的尺寸是3x3
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

#全连接层，先将前一层输出的二维特征图flatten为一维的。
#全连接有64个神经元节点,全连接到第三层卷积层的输出
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#分类
model.add(Dense(1))
model.add(Activation('sigmoid'))
# 模型编译,制定损失函数为对数损失,优化算法为rmsprop,评价指标为准确率
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# 模型训练
model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        nb_epoch=epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_samples)
#保存模型
model.save_weights('trial.h5')


