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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 第一层
# 第一个神经元  ab_c a第a个输入，第b个神经元，第c层
w11_1 = np.random.random()  # 权重：第1个输入的连接的第1个神经元，属于第1层
b1_1 = np.random.random()  # 偏置：第1个神经元，属于第1层
# 第二个神经元
w12_1 = np.random.random()  # 权重：第1个输入的连接的第2个神经元，属于第1层
b2_1 = np.random.random()  # 偏置：第2个神经元，属于第1层
# 第二层
w11_2 = np.random.random()  # 权重：第1个输入的连接的第1个神经元，属于第2层
w21_2 = np.random.random()  # 权重：第2个输入的连接的第1个神经元，属于第2层
b1_2 = np.random.random()  # 偏置：第1个神经元，属于第2层

# 前向传播
alpha = 0.01


def forward_propagation(xs):
    z1_1 = w11_1 * xs + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1 * xs + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_2, z1_2, a2_1, z2_1, a1_1, z1_1


a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(xs)

plt.plot(xs, a1_2)
plt.show()

for epoch in range(5000000):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        # Predict result through Forward Propagation
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(x)
        # Back Propagation
        # cost function e
        e = (y - a1_2) ** 2

        deda1_2 = -2 * (y - a1_2)

        da1_2dz1_2 = a1_2 * (1 - a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        #第二层的两个权重参数
        dedw11_2 = deda1_2 * da1_2dz1_2 * dz1_2dw11_2
        dedw21_2 = deda1_2 * da1_2dz1_2 * dz1_2dw21_2

        dz1_2db1_2 = 1
        dedb1_2 = deda1_2 * da1_2dz1_2 * dz1_2db1_2

        # 隐藏层的神经元
        # 隐藏层 第一个输入，第一个神经元，第一层，的权重
        dz1_2da1_1 = w11_2
        da1_1dz1_1 = a1_1 * (1 - a1_1)
        dz1_1dw11_1 = x
        dedw11_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1dw11_1
        # 隐藏层 第一个神经元，第一层 偏置
        dz1_1db1_1 = 1
        dedb1_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1db1_1

        # 隐藏层 第1个输入,第2个神经元，第1层，权重
        dz1_2da2_1 = w21_2
        da2_1dz2_1 = a2_1 * (1 - a2_1)
        dz2_1dw12_1 = x
        dz2_1db2_1 = 1
        dedw12_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1dw12_1
        # 隐藏层 第2个神经元，第1层，偏置
        dedb2_1 = deda1_2 * da1_2dz1_2 * dz1_2da2_1 * da2_1dz2_1 * dz2_1db2_1

        w11_1 = w11_1 - alpha * dedw11_1
        w11_2 = w11_2 - alpha * dedw11_2
        w12_1 = w12_1 - alpha * dedw12_1
        w21_2 = w21_2 - alpha * dedw21_2
        b1_1 = b1_1 - alpha * dedb1_1
        b2_1 = b2_1 - alpha * dedb2_1
        b1_2 = b1_2 - alpha * dedb1_2

        # differential with respect to w,b
        # z = w * x + b
        # a = 1 / (1 + np.exp(-z))
        # e = (y - a) ** 2
        # deda = -2 * (y - a)
        # dadz = a * (1 - a)
        # dzdw = x
        # dzdb = 1
        # dedw = deda * dadz * dzdw
        # dedb = deda*dadz*dzdb
        # w = w- alpha*dedw
        # b = b- alpha*dedb

    if epoch % 100 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward_propagation(xs)
        plt.plot(xs, a1_2)
        plt.pause(0.01)
