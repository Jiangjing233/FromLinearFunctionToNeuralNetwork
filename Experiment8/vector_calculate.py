import dataset
import numpy as np
import plot_utils

m = 100
X, Y = dataset.get_beans(m)
print(X, Y)
plot_utils.show_scatter(X, Y)

#W1 = 0.1
#W2 = 0.1
W=np.array([0.1,0.1])
B=np.array([0.1])

x1s = X[:, 0]  # cut the 1st row form all row
x2s = X[:, 1]  # cut the 1st column from all column



def forward_propagation(X):
    Z = X.dot(W.T) + B  #X dot-products transposed W and plus B
    A = 1 / (1 + (np.exp(-Z))) #broadcasting of each element of Z

    return A


plot_utils.show_scatter_surface(X, Y, forward_propagation)

for _ in range(1000):
    for i in range(m):
        Xi = X[i]
        Yi = Y[i]

        A = forward_propagation(Xi)
        E = (Yi - A) ** 2

        dEdA = -2 * (Yi - A)
        dAdZ = A * (1 - A)
        dZdW = Xi
        dZdB = 1

        dEdW= dEdA * dAdZ * dZdW

        dEdB = dEdA * dAdZ * dZdB
        # gradient descend
        alpha = 0.1
        W = W - alpha * dEdW

        B = B - alpha * dEdB


plot_utils.show_scatter_surface(X, Y, forward_propagation)
