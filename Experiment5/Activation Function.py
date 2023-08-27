# edited by jiangjing
import numpy as np
import matplotlib.pyplot as plt
import dataset
from mpl_toolkits.mplot3d import Axes3D

m = 100
xs, ys = dataset.get_beans(m)
plt.title("size-toxicity")
plt.xlabel("size")
plt.ylabel("toxicity")

plt.scatter(xs, ys)

w = 0.1
b = 0.1
alpha = 0.01
z = w * xs + b
a = 1 / (1 + np.exp(-z))

plt.plot(xs, a)
plt.show()

for _ in range(50000000):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # differential with respect to w,b
        z = w * x + b
        a = 1 / (1 + np.exp(-z))
        e = (y - a) ** 2
        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x
        dzdb = 1
        dedw = deda * dadz * dzdw
        dedb = deda*dadz*dzdb
        w = w- alpha*dedw
        b = b- alpha*dedb

    if _%100==0:
        plt.clf()
        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1/(1+np.exp(-z))
        plt.scatter(xs, ys)
        plt.xlim(0, 1)
        plt.ylim(0, 1.5)
        plt.plot(xs, a)
        plt.pause(0.001)
