import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())

def BaseModel():
    model = Sequential()
    model.add(Flatten(input_shape=[160, 320, 3]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=[160, 320, 3]))
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
