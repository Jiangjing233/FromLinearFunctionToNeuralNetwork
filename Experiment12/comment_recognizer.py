
from tensorflow.keras.optimizers import SGD

import shopping_data
from keras.preprocessing import sequence
from keras.layers import Embedding

import os

from keras import Sequential  # piled of NN

import keras.optimizers


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

x_train, y_train, x_test, y_test = shopping_data.load_data()
print("x_train shape", str(x_train.shape))
print("x_test shape", str(x_test.shape))
print("y_train shape", str(y_train.shape))
print("y_test shape", str(y_test.shape))
print(x_train[0])
print(y_train[0])

vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)
print(word_index)
print("词典总词语数量", vocalen)

x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)

maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen=maxlen)

model = Sequential()
model.add(Embedding(trainable=False, input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(keras.layers.Flatten())
dense = keras.layers.Dense(units=256, activation="relu")  # unit: the number of Neurons in current layer is12
model.add(dense)
dense = keras.layers.Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = keras.layers.Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
dense = keras.layers.Dense(units=256, activation='relu')  # unit: the number of Neurons in current layer
model.add(dense)
# the sum of the output is 1, p= single sigmoid each output/sum of all sigmoid output
dense = keras.layers.Dense(units=1, activation='sigmoid')  # unit: the number of Neurons in current layer
model.add(dense)

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])

model.fit(x_train_index, y_train, epochs=5000, batch_size=256)

score, accuracy = model.evaluate(x_test_index, y_test)
print("Score", str(score))
print("accuracy", str(accuracy))
