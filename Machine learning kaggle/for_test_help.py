import tensorflow as tf
import numpy as np
import pandas as pd

column_names = ['code number', \
                'Clump Thickness', \
                'Uniformity of Cell Size', \
                'Uniformity of Cell Shape', \
                'Marginal Adhesion', \
                'Single Epithelial Cell Size', \
                'Bare Nuclei', \
                'Bland Chromatin', \
                'Normal Nucleoli', \
                'Mitoses', \
                'class_label']
dataset =pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
# print(dataset)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(dataset[['Clump Thickness', 'Uniformity of Cell Size']], dataset['class_label'], test_size=0.25, random_state=33)
x_train = np.float32(x_train.T)
x_test = np.float32(x_test.T)
y_train = np.float32(y_train.T)
y_test = np.float32(y_test.T)
# print(x_train,y_train)

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random.uniform([1,2],-1.0,1.0))
y = tf.matmul(w,x_train)+b
loss = tf.reduce_mean(tf.square(y-y_train))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
tf.init()

for i in range(1000):
    tf.train()
    if i%200==0:
        tf.print('i:',i,'\tw:',w,'\tb:',b'\tloss:',loss)



