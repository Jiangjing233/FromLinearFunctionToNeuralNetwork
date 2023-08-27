# le-net5
import os
from tensorflow.keras.datasets import mnist
#import numpy as np
from tensorflow.keras.models import Sequential  # piled of NN
from tensorflow.keras.layers import Dense  # full connected layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
#import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# training set, label set, test set, label set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# devided by 255.0 to squished shape of scale
X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0
# m=n-f+1, n:original size, f:foot of each convolution it moves, m:New size

# one-hot conversion, the label of training and test set
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()
#filter: the core number of Convolutional Kernel
#kernel_size: size
#strides: move 1 step after each operation, and upward 1 step for after each row
#input_shape: the shape of the input
# padding : sets as valid meaning that not using 0 to fill with blank
model.add(
    Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu'))
# subsampling by 2*2 Max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
# subsampling by 2*2 Max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#flatten the data to seed into full connected layer
model.add(Flatten())

dense = Dense(units=120, activation='relu')  # unit: the number of Neurons in current layer is12
model.add(dense)
dense = Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = Dense(units=84, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
# the sum of the output is 1, p= single sigmoid each output/sum of all sigmoid output
dense = Dense(units=10, activation='softmax')  # unit: the number of Neurons in current layer
model.add(dense)

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5000, batch_size=4096)

loss, accuracy = model.evaluate(X_test, Y_test)
print("accuracy", str(accuracy))
print("loss", str(loss))
