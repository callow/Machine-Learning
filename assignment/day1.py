#这一步不知道是做什么的
print(__doc__)
# 导入相关包
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# 加载糖尿病的数据集
diabetes = datasets.load_diabetes()

#print(diabetes)
# 这里只用了一个feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

#print(diabetes_X)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 创建一个回归模型的object
regr = linear_model.LinearRegression()

# 用训练集创建模型+训练模型
regr.fit(diabetes_X_train, diabetes_y_train)

# 打印 coefficients / 权值 ，Y = ax +b  , a = 938.23...
print('Coefficients: \n', regr.coef_)
# 打印 mean square error /均方误
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# 打印评分函数，将返回一个小于1的得分: 1 is perfect prediction， 不过不知道这一步是做什么的？
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# 画出样本散点(x,y)
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# 推算并画出此线性方程直线 用蓝色标出
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)
         

#这两步是做什么的？
plt.xticks(())
plt.yticks(())

plt.show()