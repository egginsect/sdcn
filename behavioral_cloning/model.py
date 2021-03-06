import os
import tensorflow as tf
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

def BaseModel():
    model = Sequential()
    model.add(Flatten(input_shape=[160, 320, 3]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def LeNet():
    model = Sequential()
    model.add(Cropping2D(cropping=((40,0),(0,0)), input_shape=[160, 320, 3]))
    #model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=[160, 320, 3]))
    model.add(Lambda(lambda x: x/255.0 -0.5))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='selu'))
    model.add(Conv2D(64, (3, 3), activation='selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(84, activation='selu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
