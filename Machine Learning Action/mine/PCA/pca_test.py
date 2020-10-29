# -*- coding: utf-8 -*-

# @File    : pca_test.py
# @Date    : 2020-10-28
# @Author  : YUEYUE-x4
# @Demo    :  
print('PCA test!')

#%%
print('PCA test begin!')
import PCA
import imp
imp.reload(PCA)
import numpy as np

datamat = PCA.load_dataset('testSet.txt')
# print(datamat)
lower_mat,recon_mat = PCA.pca(datamat,1)


print('PCA test end!')