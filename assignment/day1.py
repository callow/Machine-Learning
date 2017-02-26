#��һ����֪������ʲô��
print(__doc__)
# ������ذ�
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# �������򲡵����ݼ�
diabetes = datasets.load_diabetes()

#print(diabetes)
# ����ֻ����һ��feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

#print(diabetes_X)
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# ����һ���ع�ģ�͵�object
regr = linear_model.LinearRegression()

# ��ѵ��������ģ��+ѵ��ģ��
regr.fit(diabetes_X_train, diabetes_y_train)

# ��ӡ coefficients / Ȩֵ ��Y = ax +b  , a = 938.23...
print('Coefficients: \n', regr.coef_)
# ��ӡ mean square error /������
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# ��ӡ���ֺ�����������һ��С��1�ĵ÷�: 1 is perfect prediction�� ������֪����һ������ʲô�ģ�
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# ��������ɢ��(x,y)
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# ���㲢���������Է���ֱ�� ����ɫ���
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)
         

#����������ʲô�ģ�
plt.xticks(())
plt.yticks(())

plt.show()