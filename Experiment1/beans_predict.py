import dataset
from matplotlib import pyplot as plt

xs, ys = dataset.get_beans(100)
print(xs, ys)
# 配置坐标系
plt.title("size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size")  # title of X-axis
plt.ylabel("Toxic")  # title of Y - axis
plt.scatter(xs, ys)

w = 0.5 # weight
alpha = 0.05 # rate of study
for j in range(100):
    for i in range(100):#迭代了一次
        x = xs[i]
        y = ys[i]
        y_pre = w * x  # xs is array, it can be operated respectively
        e = y - y_pre
        w = w + alpha * e * x

y_pre = w * xs

plt.plot(xs, y_pre)

plt.show()  # refresh of the graph
