from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D

# 读入数据
data = genfromtxt(r"delivery.csv", delimiter=",")
# print(data)
x = data[:, :-1]  # 前两列
y = data[:, -1]  # 最后一列

model = LinearRegression()
model.fit(x, y)

# 系数
# 维数和特征数一致
print("coefficient={0}".format(model.coef_))

# 截距
print("intercept={0}".format(model.intercept_))

# 预测值
test = [[102, 4]]
predict = model.predict(test)
print("predict={0}".format(predict))

# 画3D图形
fig1 = plt.figure()
ax = Axes3D(fig1)
ax.scatter(x[:, 0], x[:, 1], y, marker="o", c='r', s=100)
x0 = x[:, 0]
x1 = x[:, 1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.coef_[0] * x0 + model.coef_[1] * x1 + model.intercept_
ax.plot_surface(x0, x1, z)
ax.set_xlabel('miles')
ax.set_ylabel('number of deliveries')
ax.set_zlabel('time')
# 显示
plt.show()
