# -*- coding: utf-8 -*-

# @File    : regression_test.py
# @Date    : 2020-10-24
# @Author  : YUEYUE-x4
# @Demo    :  


#%%
print('Standard regression test begin!')
import REGRESSION
import imp
imp.reload(REGRESSION)
import numpy as np
X,Y = REGRESSION.load_dataset('ex0.txt')
ws = REGRESSION.standard_regression(X,Y)
Y_predict = np.mat(X)*ws
# 计算相关系数
corr = np.corrcoef(Y,Y_predict.T.tolist())
print("相关系数：",corr)
import matplotlib.pyplot as plt
plt.figure(1,figsize=[10,6])
plt.scatter(np.mat(X)[:,1].tolist(),Y)
plt.scatter(np.mat(X)[:,1].tolist(),Y_predict.tolist())
plt.show()
print('Standard regression test end!')

#%%
print('LWLR regression test begin!')
# 局部加权回归
import REGRESSION
import imp
imp.reload(REGRESSION)
import numpy as np
X,Y = REGRESSION.load_dataset('ex0.txt')
X_test = X
Y_predict = REGRESSION.lwlr_fit(X_test,X,Y,k=0.02)
# 计算相关系数
corr = np.corrcoef(Y,Y_predict.T)
print("相关系数：",corr)


# list排序
X_list = np.mat(X)[:,1].T.tolist()[0]
plot_index = list(np.argsort(X_list))
plot_x = [X_list[index_i] for index_i in plot_index]
plot_y = [Y_predict[index_i] for index_i in plot_index]

# plot_x = np.mat(X)[plot_index][:,0,1].T.tolist()[0]
# plot_y = np.mat(Y_predict).T[plot_index][:,0,0].T.tolist()[0]
# print('plot x: ',plot_y)
# plt.scatter(np.mat(X)[:,1].tolist(),Y)
import matplotlib.pyplot as plt
plt.figure(1,figsize=[10,6])
plt.scatter(np.mat(X)[:,1].tolist(),Y)
plt.plot(plot_x,plot_y)
plt.show()

print('LWLR regression test end!')

#%%
print('ridge regression test begin!')
# 局部加权回归
import REGRESSION
import imp
imp.reload(REGRESSION)
import numpy as np
X,Y = REGRESSION.load_dataset('abalone.txt')
Y_fit = REGRESSION.ridge_fit(X,X,Y,lamb=0.002)

corr = np.corrcoef(Y, Y_fit.T.tolist()[0])
print(corr)

import matplotlib.pyplot as plt
plt.figure(1,figsize=[10,6])
dimension = 1
# plt.scatter(np.mat(X)[:,dimension].tolist(),Y)
# plt.scatter(np.mat(X)[:,dimension].tolist(),Y_fit[:,0].tolist())
plt.plot(Y, '.')
plt.plot(Y_fit.T.tolist()[0], '.')
plt.legend(['origin','fit'])
# plt.ylim([-2000,2000])
plt.show()

print('ridge regression test end!')

#%%
print('stage wise regression test begin!')
# 局部加权回归
import REGRESSION
import imp
imp.reload(REGRESSION)
import numpy as np
X,Y = REGRESSION.load_dataset('abalone.txt')
ws_mat = REGRESSION.stage_wise(X,Y,0.001,5000)

import matplotlib.pyplot as plt
plt.figure(1,figsize=[10,6])
plt.plot(ws_mat)
plt.savefig('stage wise.png',dpi=2400)
plt.show()

print('stage wise regression test end!')