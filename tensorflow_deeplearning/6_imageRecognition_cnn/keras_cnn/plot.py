# coding:utf-8
from keras.models import Sequential
from keras.utils.visualize_util import plot
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

print("Now we build the model")
model = Sequential()
img_width, img_height = 150, 150
# img_channels = 3 #output dimenson nothing with channels
# img_rows = 150
# img_cols = 150
# model.add(Convolution2D(32, 3, 3, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', dim_ordering='th',input_shape=(img_channels,img_rows,img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', dim_ordering='th'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', dim_ordering='th'))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
# model.add(Activation('relu'))
# model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
#
# adam = Adam(lr=1e-6)
# model.compile(loss='mse',optimizer=adam)


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
print("We finish building the model")

plot(model, to_file='model3.png', show_shapes=True)