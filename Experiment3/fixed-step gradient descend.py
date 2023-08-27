import numpy as np
import matplotlib.pyplot as plt
import dataset

# generate data
xs, ys = dataset.get_beans(100)

# Configurate graph
plt.title("size-toxicity function", fontsize=12)
plt.xlabel("bean-size")
plt.ylabel("toxicity")
plt.scatter(xs, ys)

# initialize value of w
w = 0.1
y_pre = w * xs  # prediction of xs and toxicity relationship

plt.plot(xs, y_pre)
plt.show()
alpha = 0.001
step = 0.01
for _ in range(1000):
    for i in range(100): # a epoch of training
        x = xs[i]
        y = ys[i]  # 2rd coefficient : x0*x0, 1st coefficient :-2x0*y0 constant term:y^2
        # slope : 2aw+b = 2*xs*xs*w - 2xs*ys
        k = 2 * np.sum(x ** 2) * w - 2 * np.sum(x) * np.sum(y) #there are 100 times of k sum, so devided by100
        k= 0.01*k
        if k>0:
           w = w - step
        else:
            w = w + step
        plt.clf() #clear the gram
        y_pre= w*xs #update the W into prediction curve
        plt.scatter(xs,ys)
        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs, y_pre)
        plt.pause(0.01)


#redraw the graph
# plt.scatter(xs,ys)
# y_pre = w*xs
# plt.plot(xs,y_pre)
# plt.show()

# draw a dynamic graph
