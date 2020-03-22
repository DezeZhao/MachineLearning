"""
一元线性回归——梯度下降法
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 最小二乘法(损失函数/代价函数)
def loss_func(k, b, x, y):
    # totalError = 0
    # for i in range(0, len(x)):
    #     totalError += (y[i] - (k * x[i] + b)) ** 2
    # return totalError / float(len(x)) / 2.0
    return 1 / 2 * np.mean((k * x + b - y) ** 2)


# 可视化损失函数 生成网格x,y 对应的点z
def loss_mesh(k, b, x, y):
    z = np.zeros((k.shape[0], k.shape[1]))
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            z[i, j] = loss_func(k[i, j], b[i, j], x, y)
    return z


# 梯度下降法
def gradient_descent(x, y, b, k, lr, epochs, ax):
    # 总数据量
    m = float(len(x))
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        # 对k , b求偏导
        for j in range(0, len(x)):
            b_grad += (1 / m) * (k * x[j] + b - y[j])
            k_grad += (1 / m) * (k * x[j] + b - y[j]) * x[j]
        # k_grad = np.mean((k * x + b - y) * x)
        # b_grad = np.mean(k * x + b - y)
        # 同步更新 k b
        k -= lr * k_grad
        b -= lr * b_grad
        # 绘制当前参数所在的位置 每59步绘制一次
        if i % 50 == 0:
            ax.scatter3D(k, b, loss_func(k, b, x, y), marker='o', s=30, c='blue')
    return k, b


if __name__ == '__main__':
    lr = 0.01  # 学习率
    k = 0  # 斜率
    b = 0  # 截距
    epochs = 600  # 最大步数
    # 载入数据
    # data = np.genfromtxt("data.txt", delimiter=",")
    np.random.seed(1)
    x = np.arange(-1, 1, step=0.04)  # 自变量
    noise = np.random.uniform(low=-0.5, high=0.5, size=50)  # 噪声
    y = x * 3 + 2 + noise  # 因变量

    k_axis = np.linspace(start=0, stop=5, num=50)
    b_axis = np.linspace(start=0, stop=5, num=50)
    k_m, b_m = np.meshgrid(k_axis, b_axis)  # 生成网格点矩阵
    z_m = loss_mesh(k_m, b_m, x, y)
    # 绘制三维图像
    fig1 = plt.figure(num="fig1")
    ax1 = Axes3D(fig1)
    ax1.set_xlabel('k')
    ax1.set_ylabel('b')
    ax1.set_zlabel('loss')
    ax1.plot_surface(k_m, b_m, z_m, rstride=1, cstride=1, cmap=plt.cm.hot, alpha=0.5)  # 绘制3D面，rstride 行跨度，cstride 列跨度
    k, b = gradient_descent(x, y, b, k, lr, epochs, ax1)
    print("k={0},b={1}".format(k, b))
    # 绘制原始训练集  和 回归曲线
    plt.figure(num="fig2")  # num用于标识区分多个figure
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'bo', x, k * x + b, 'r-')
    plt.show()
    plt.close()
