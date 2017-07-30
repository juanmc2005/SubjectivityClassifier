import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# For reproducibility
np.random.seed(123)
# set to use channel first (i.e. (1, 28, 28) instead of (28, 28, 1))
K.set_image_dim_ordering('th')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape from (n, width, height) to (n, depth, width, height)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

# Values as float32 and normalized
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# To one-hot vector format
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

# Evaluation
score = model.evaluate(x_test, y_test, verbose=0)

print(score)
