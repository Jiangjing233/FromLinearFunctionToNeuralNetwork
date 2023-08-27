import dataset
import numpy as np
import plot_utils
from keras.models import Sequential #piled of NN
from keras.layers import Dense # full connected layer

# Single neuron classification model


m=100
X,Y=dataset.get_beans1(m)
plot_utils.show_scatter(X,Y)
model=Sequential()
dense=Dense(units=1, activation='sigmoid', input_dim=1)  #unit: the number of Neurons in current layer
model.add(dense)

model.compile(loss='mean_squared_error',optimizer='sgd', metrics=['accuracy'])
model.fit(X,Y,epochs=5000, batch_size=10)

pres=model.predict(X)

plot_utils.show_scatter_curve(X,Y,pres)

# hidden layer neurons classification model

