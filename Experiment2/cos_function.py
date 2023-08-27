import dataset
from matplotlib import pyplot as plt
import numpy as np

# generate data for the bean
xs, ys = dataset.get_beans(100)
print(xs, ys)  # they are numpy arraies

# plot the graph of size - toxicity
plt.title("size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")  # title of X-axis
plt.ylabel("Toxic")  # title of Y - axis
plt.scatter(xs, ys)

# initialize the slope of y_pre
w = 0.1
m = 100

y_pre = w * xs
plt.plot(xs, y_pre)
plt.show()

# cost_function array of every single bean, m is the number of the data
es = (ys - y_pre) ** 2

# total cost  = sum of every single bean
sum_e = np.sum(es)
sum_e = (1 / m) * sum_e
print(sum_e)
ws = np.arange(0, 3, 0.1)  # start,end,step
es = []

for w in ws:
    y_pre = w * xs
    e = (1 / m) * (np.sum((ys - y_pre) ** 2))
    print("w:" + str(w) + "e:" + str(e))
    es.append(e)

# config the graph
plt.title("Cos吧"
          "t Function", fontsize=12)
plt.xlabel("w")  # title of X-axis
plt.ylabel("e")  # title of Y - axis
plt.plot(ws, es)
plt.show()

#  minimul point of cost funcion
w_min = np.sum(xs * ys) / np.sum(xs * xs)
print("the minimum point of w is  " + str(w_min))

plt.show()
# new w iterated into y
y_pre = w_min * xs


plt.title("size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")  # title of X-axis
plt.ylabel("Toxic")  # title of Y - axis
plt.scatter(xs, ys)
plt.plot(xs, y_pre)
plt.show()
