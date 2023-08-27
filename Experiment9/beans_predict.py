import dataset
import numpy as np
import plot_utils
from keras.models import Sequential #piled of NN
from keras.layers import Dense # full connected layer
from keras.optimizers import SGD

m=100
X,Y=dataset.get_beans(m)
plot_utils.show_scatter(X,Y)

model=Sequential()
dense=Dense(units=8, activation='relu', input_dim=2)  #unit: the number of Neurons in current layer
model.add(dense)

dense=Dense(units=8, activation='relu')  #unit: the number of Neurons in current layer
model.add(dense)
dense=Dense(units=8, activation='relu')  #unit: the number of Neurons in current layer
model.add(dense)
dense=Dense(units=8, activation='relu')  #unit: the number of Neurons in current layer
model.add(dense)
dense=Dense(units=1, activation='sigmoid')  #unit: the number of Neurons in current layer
model.add(dense)


model.compile(loss='mean_squared_error',optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X,Y,epochs=50000, batch_size=100)

pres=model.predict(X)

plot_utils.show_scatter_surface(X,Y,model)
print(model.get_weights())