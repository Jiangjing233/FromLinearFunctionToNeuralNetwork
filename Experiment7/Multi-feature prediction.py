import dataset
import numpy as np
import plot_utils

m = 100
xs, ys = dataset.get_beans(m)
print(xs, ys)
plot_utils.show_scatter(xs, ys)

w1 = 0.1
w2 = 0.1
b = 0.1

x1s = xs[:,0]  # cut the 1st row form all row
x2s = xs[:,1]  # cut the 1st column from all column


def forward_propagation(x1s, x2s):
    z = w1 * x1s + w2 * x2s + b
    a = 1 / (1 + (np.exp(-z)))
    return a


plot_utils.show_scatter_surface(xs,ys,forward_propagation)

for _ in range(500000):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        x1 = x[0]
        x2 = x[1]
        a = forward_propagation(x1,x2)
        e = (y - a) ** 2

        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1
        dedw1 = deda * dadz * dzdw1
        dedw2 = deda * dadz * dzdw2
        dedb = deda * dadz * dzdb
        # gradient descend
        alpha = 0.1
        w1 = w1 - alpha * dedw1
        w2 = w2 - alpha * dedw2
        b = b - alpha * dedb


plot_utils.show_scatter_surface(xs,ys,forward_propagation)
