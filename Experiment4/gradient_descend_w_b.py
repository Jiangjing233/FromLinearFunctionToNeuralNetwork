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
y_pre = w * xs + b

plt.plot(xs, y_pre)
plt.show()

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        dw = 2 * (x ** 2) * w - 2 * x * y + 2 * x * b #a epoch of gradient descend
        db = 2*b + 2*(x*w-y)
        w = w - alpha * dw
        b = b -alpha* db
        plt.clf()
        plt.scatter(xs, ys)
        y_pre = w * xs +b
    plt.clf()
    plt.scatter(xs, ys)
    plt.xlim(0,1)
    plt.ylim(0,1.5)
    plt.plot(xs,y_pre)
    plt.pause(0.01)



