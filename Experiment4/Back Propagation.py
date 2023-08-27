# edited by jiangjing
# import libs
import numpy as np
import matplotlib.pyplot as plt
import dataset
from mpl_toolkits.mplot3d import Axes3D

m = 100
xs, ys = dataset.get_beans(m)
plt.title("Size-toxicity function", fontsize=12)
plt.xlabel("Bean-size")
plt.ylabel("toxicity")
plt.xlim(0, 1)
plt.ylim(0, 1.3)
plt.scatter(xs, ys)
w = 0.1
b = 0.1
y_pre = w * xs + b
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(0, 2)


ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.1)
for b in bs:
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = np.sum((ys - y_pre)**2)*(1/m)
        es.append(e)
    ax.plot(ws, es, b, zdir='y')

plt.show()
