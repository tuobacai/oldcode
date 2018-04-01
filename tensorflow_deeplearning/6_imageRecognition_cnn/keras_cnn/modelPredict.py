from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import os
img_width, img_height = 150, 150
train_data_dir = '/Users/yingjie10/Documents/data/train'
validation_data_dir = '/Users/yingjie10/Documents/data/validation'
test_data_dir = '/Users/yingjie10/Documents/data/test'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50
def load_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height),dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

    model.load_weights('/Users/yingjie10/PycharmProjects/imageRecognition/keras_cnn/trial.h5')
    # model.load_weights('/Users/yingjie10/PycharmProjects/imageRecognition/keras_cnn/trial_50_4h.h5')
    return model
if __name__=='__main__':
    model=load_model()
    out=open('res_predict1.txt','w')
    out.write("id,predict\n")
    for i in range(1,13):
        path = os.path.join(test_data_dir, str(i) + '.jpg')
        im = Image.open(path)
        img = im.resize((img_width, img_height))
        x = img_to_array(img, dim_ordering='th')  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        score = model.predict_classes(x, batch_size=16,verbose=1)
        out.write( str(i)+","+str(score[0][0])+"\n")
        out.flush()
    out.close()