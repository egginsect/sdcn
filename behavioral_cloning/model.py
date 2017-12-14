from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def BaseModel():
    model = Sequential()
    model.add(Flatten(input_shape=[160, 320, 3]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def LeNet():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', 
    input_shape=[160, 320, 3]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')
    return model
