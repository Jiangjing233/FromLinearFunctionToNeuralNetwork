import os
from tensorflow.keras.datasets import mnist

import numpy as np
from tensorflow.keras.models import Sequential  # piled of NN
from tensorflow.keras.layers import Dense  # full connected layer
from tensorflow.keras.optimizers import SGD
#import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# training set, label set, test set, label set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("x_train.shape" + str(X_train.shape))
print("y_train.shape" + str(Y_test.shape))  # classification result or label of 60000 number
print("x_test.shape" + str(X_train.shape))
print("y_test.shape" + str(Y_test.shape))

#show the 1st image of training set as grayscale
#plt.imshow(X_train[0], cmap='gray')  # plot X_train as Grayscale graph
#out put the label
#print(Y_train[0])
#plt.show()

#devided by 255.0 to squished shape of scale
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0

# one-hot conversion, the label of training and test set
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()
dense = Dense(units=256, activation='relu', input_dim=784)  # unit: the number of Neurons in current layer
model.add(dense)

dense = Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
#the sum of the output is 1, p= single sigmoid each output/sum of all sigmoid output
dense = Dense(units=10, activation='softmax')  # unit: the number of Neurons in current layer
model.add(dense)

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5000000, batch_size=2048)

for epochs in range(5000000):
   # plt.clf()

   # plt.ylim(0, 1)
  #  plt.plot(epochs, accuracy)
   # plt.pause(0.01)
    loss, accuracy = model.evaluate(X_test, Y_test)
  #  plt.show()
    print("accuracy", str(accuracy))
    print("loss", str(loss))
