# -*- coding: utf-8 -*-

# @File    : svd_test.py
# @Date    : 2020-10-28
# @Author  : YUEYUE-x4
# @Demo    :

print('SVD test!')

#%%
print('SVD test begin!')
import SVD
import imp
imp.reload(SVD)
import numpy as np

test_flag = 0

if test_flag==0:
    datamat = SVD.load_ex_data()
    u,sigma,vt = np.linalg.svd(datamat)
    # print(u,sigma,vt)
    nums = 3
    sigma_nums = np.mat(np.diag(sigma[:nums]))
    u_nums = u[:,:nums]
    vt_nums = vt[:nums, :]
    datamat_nums = u_nums*sigma_nums*vt_nums
    print(np.max(np.abs(datamat_nums-datamat)))

print('SVD test end!')


