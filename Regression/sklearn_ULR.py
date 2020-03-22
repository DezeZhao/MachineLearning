from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x = np.arange(-1, 1, step=0.04)  # 自变量
noise = np.random.uniform(low=-0.5, high=0.5, size=50)  # 噪声
y = x * 3 + 2 + noise  # 因变量

# 此处必须为二维数组 转换方法如下
# a = np.array([1,2,3]);
# b = np.array([[1],[2],[3]]);
# #将一维数组a转化为二维数组
# a = a[:,np.newaxis];

data = list((x, y))
x = x[:, np.newaxis]
y = y[:, np.newaxis]
# 创建并拟合模型
model = LinearRegression()
model.fit(x, y)

plt.figure()
plt.plot(x, y, 'bo', x, model.predict(x), 'r-')
plt.show()
