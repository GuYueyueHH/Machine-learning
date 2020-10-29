# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
print('sklearn test!')


# %%
print('LR and SGD test begin!')
# logistic regression
# SGD 分类
# 乳腺癌肿瘤预测

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

column_names = ['code number','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','class_label']
# dataset =pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
# 导入数据
dataset = pd.read_csv('./dataset/breast-cancer-wisconsin.csv')
dataset = dataset.replace('?',np.nan)
# print('dataset shape:',dataset.shape)
dataset = dataset.dropna(how='any')
# print('dataset shape:',dataset.shape)

# cross_validation 交叉检验
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(dataset[column_names[1:9]], dataset[column_names[-1]], test_size=0.25, random_state=33)
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 归一化处理 standardscaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
# print(x_train)
x_train = ss.fit_transform(x_train)
x_test  = ss.transform(x_test)
# print(x_train)

# LogisticRegression和SGDclassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 实例化
lr = LogisticRegression()
sgdc = SGDClassifier()

# 模型拟合
lr.fit(x_train,y_train)
sgdc.fit(x_train,y_train)

# 预测
lr_y_predict = lr.predict(x_test)
sgdc_y_predict = sgdc.predict(x_test)

# 模型报告
from sklearn.metrics import classification_report
lr_report = classification_report(y_test, lr_y_predict, target_names=['2','4'])
sgdc_report = classification_report(y_test, sgdc_y_predict, target_names=['2','4'])
print('lr--report:\n',lr_report)
print('sgdc--report:\n',sgdc_report)

print('LR and SGD test end!')


# %%
print('SVC test begin!')
import pandas as pd

from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)

# 数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# print(x_train.shape,x_test.shape)

# 标准化处理
# print(pd.unique(x_train[:,1]))
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 模型拟合
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(x_train,y_train)
y_predict = lsvc.predict(x_test)

# 评估
print('linear svc test score:',lsvc.score(x_test,y_test))
print('linear svc train score:',lsvc.score(x_train,y_train))
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))      
print('\nlinear svc classification report:\n',cr)


print('SVC test end!')


# %%
print('Bayes test begin!')
# 朴素贝叶斯

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
# print(len(news.data))
# print(news.data[0])

# 数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)
# print(len(x_train),len(x_test))

# 文本特征向量转换模块
from sklearn.feature_extraction.text import CountVectorizer
cvect = CountVectorizer()
x_train = cvect.fit_transform(x_train)
x_test = cvect.transform(x_test)
# print(type(x_train))

# 模型拟合
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_predict = mnb.predict(x_test)

# 模型检验
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_predict,target_names=news.target_names)
print('\nnaive bayes test report:\n',cr)


print('Bayes test end!')


# %%
print('KNN test begin!')
# KNN 分类

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.data.shape)
# print(iris.DESCR)

# 数据分割
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
# print(x_train.shape,x_test.shape)

# 数据预处理
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 模型拟合
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)

# classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_predict,target_names=iris.target_names)
print('\nknn classification report:\n',cr)

print('KNN test end!')


# %%
print('Trees test begin!')
import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(titanic.info())
# print(titanic.describe())
x = titanic[['pclass','age','sex']]
y = titanic['survived']

# age 缺失值处理
x['age'].fillna(x['age'].mean(),inplace=True)
# pclass sex处理
# print(x_train.head())
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
x = dvec.fit_transform(x.to_dict(orient='record'))
# x = pd.DataFrame(x)
# print(x.describe())

# crossvalidation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=33)
# print(x_train.shape,x_test.shape)

# 模型拟合
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_predict = dtree.predict(x_test)

# classificaton report
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_predict,target_names=['died','survived'])
print('\ndecision tree classifier report:\n',cr)




print('Trees test end!')


# %%
print(type(x))


