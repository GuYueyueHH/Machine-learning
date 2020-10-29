# -*- coding: utf-8 -*-

# @File    : PCA.py
# @Date    : 2020-10-28
# @Author  : YUEYUE-x4
# @Demo    :  

import numpy as np

def load_dataset(filename,delim='\t'):
    f = open(filename)
    string_array = [line.strip().split(delim) for line in f.readlines()]
    data_array = [list(map(float,line)) for line in string_array]
    f.close()
    return np.mat(data_array)

def pca(datamat,top_n_feature=9999999):
    mean_values = np.mean(datamat,axis=0)
    mean_removed = datamat - mean_values
    cov_mat = np.cov(mean_removed,rowvar=0)
    eigen_values,eigen_vectors = np.linalg.eig(np.mat(cov_mat))
    eigen_values_index = np.argsort(eigen_values)
    eigen_values_index = eigen_values_index[:-(top_n_feature+1):-1]
    selected_eigen_vectors = eigen_vectors[:,eigen_values_index]
    lower_datamat = mean_removed*selected_eigen_vectors
    recon_mat = (lower_datamat*selected_eigen_vectors.T)+mean_values
    print(eigen_vectors)
    return lower_datamat,recon_mat
