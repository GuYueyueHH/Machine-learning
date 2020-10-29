# -*- coding: utf-8 -*-

# @File    : SVD.py
# @Date    : 2020-10-28
# @Author  : YUEYUE-x4
# @Demo    :  

import numpy as np

def load_ex_data():
    datamat = [[0, 0, 0, 2, 2],
               [0, 0, 0, 3, 3],
               [0, 0, 0, 1, 1],
               [1, 1, 1, 0, 0],
               [2, 2, 2, 0, 0],
               [5, 5, 5, 0, 0],
               [1, 1, 1, 0, 0]]
    return np.mat(datamat)