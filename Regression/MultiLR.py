"""
多元线性回归
梯度下降法
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 最小二乘法 代价函数 损失函数
def loss_func(w0, w1, w2, x, y):
    return np.mean((x[:, 0] * w1 + x[:, 1] * w2 + w0 - y) ** 2)


# 梯度下降法
def gradient_descent(x, y, w0, w1, w2, epochs, lr):
    m = float(len(x))
    for i in range(epochs):
        w0_gard = 0
        w1_gard = 0
        w2_gard = 0
        # for j in range(0, len(x)):
        #     w0_gard += (1 / m) * (x[j, 0] * w1 + x[j, 1] * w2 + w0 - y[j])
        #     w1_gard += (1 / m) * (x[j, 0] * w1 + x[j, 1] * w2 + w0 - y[j]) * x[j, 0]
        #     w2_gard += (1 / m) * (x[j, 0] * w1 + x[j, 1] * w2 + w0 - y[j]) * x[j, 1]
        w0_gard += np.mean(x[:, 0] * w1 + x[:, 1] * w2 + w0 - y)
        w1_gard += np.mean((x[:, 0] * w1 + x[:, 1] * w2 + w0 - y) * x[:, 0])
        w2_gard += np.mean((x[:, 0] * w1 + x[:, 1] * w2 + w0 - y) * x[:, 1])
        w0 -= w0_gard * lr;
        w1 -= w1_gard * lr;
        w2 -= w2_gard * lr;
        # print(w0, w1, w2)
    return w0, w1, w2


def main():
    # 读入数据
    data = genfromtxt(r"delivery.csv", delimiter=",")
    # print(data)
    x = data[:, :-1]  # 前两列
    y = data[:, -1]  # 最后一列
    # print(x[:,0])
    # print(y)
    # learn rate
    lr = 0.0001
    # 参数
    w0 = 0
    w1 = 0
    w2 = 0
    # 迭代步数
    epochs = 1000
    print("starting w0={0},w1={1},w2={2},error={3}".format(w0, w1, w2, loss_func(w0, w1, w2, x, y)))
    print("running...")
    w0_, w1_, w2_ = gradient_descent(x, y, w0, w1, w2, epochs, lr)
    print("w0={0},w1={1},w2={2},error={3}".format(w0_, w1_, w2_, loss_func(w0_, w1_, w2_, x, y)))
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    x0 = x[:, 0]
    x1 = x[:, 1]
    # 生成网格矩阵
    x0, x1 = np.meshgrid(x0, x1)
    z = w1 * x0 + w2 * x1 + w0
    ax.plot_surface(x0, x1, z)
    ax.set_xlabel('miles')
    ax.set_ylabel('number of deliveries')
    ax.set_zlabel('time')
    # 显示
    plt.show()


if __name__ == '__main__':
    main()
