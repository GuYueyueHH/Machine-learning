# -*- coding: utf-8 -*-

# @File    : REGRESSION.py
# @Date    : 2020-10-24
# @Author  : YUEYUE-x4
# @Demo    :  

import numpy as np

def load_dataset(filename):
    num_features = len(open(filename).readline().split('\t')) -1
    data_mat = []
    labels_mat = []
    f = open(filename)
    for line in f.readlines():
        line_array = []
        cur_line = line.strip().split('\t')
        for i in range(num_features):
            line_array.append(float(cur_line[i]))
        data_mat.append(line_array)
        labels_mat.append(float(cur_line[-1]))
    f.close()
    return data_mat,labels_mat

def standard_regression(X,Y):
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).T
    xTx = np.dot(X_mat.T,X_mat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (X_mat.T*Y_mat)
    return ws

def lwlr(testpoint,X,Y,k=1.0):
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).T
    m = len(X)
    weights = np.mat(np.eye(m))
    for i in range(m):
        diff_mat = testpoint - X_mat[i,:]
        weights[i,i] = np.exp(diff_mat*diff_mat.T/(-2.0*k**2))
    xTx = X_mat.T*(weights*X_mat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (X_mat.T*(weights*Y_mat))
    return testpoint*ws

def lwlr_fit(X_test,X,Y,k=1.0):
    m = len(X_test)
    y_fit = np.zeros(m)
    for i in range(m):
        y_fit[i] = lwlr(X_test[i],X,Y,k)
    return y_fit

def ridge_regression(X,Y,lamb=0.2):
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).T
    xTx = X_mat.T*X_mat
    denom = xTx + np.eye(np.shape(X_mat)[1])*lamb
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse!')
    ws = denom.I * (X_mat.T*Y_mat)
    return ws

def ridge_fit(X_test,X,Y,lamb=0.2):
    ws = ridge_regression(X,Y,lamb)
    y_fit = np.mat(X_test)*ws
    return y_fit

def ridge_test(X,Y):
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).T
    Y_mean = np.mean(Y_mat)
    Y_mat = Y_mat - Y_mean
    X_means = np.mean(X_mat,0)
    X_var = np.var(X_mat,0)
    X_mat = (X_mat-X_means)/X_var
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts,np.shape(X_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regression(X_mat,Y_mat.T,np.exp(i-10))
        w_mat[i,:] = ws.T
    # print(ws)
    return w_mat

def regularize(X_mat):
    in_mat = X_mat.copy()
    in_means = np.mean(in_mat,0)
    in_var = np.var(in_mat,0)
    in_mat = (in_mat-in_means)/in_var
    return in_mat

def stage_wise(X,Y,eps=0.01,num_iter=100):
    X_mat = np.mat(X)
    Y_mat = np.mat(Y).T
    Y_mean = np.mean(Y_mat)
    Y_mat = Y_mat - Y_mean
    X_mat = regularize(X_mat)
    m,n = np.shape(X_mat)
    return_mat = np.zeros([num_iter,n])
    ws = np.zeros([n,1])
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(num_iter):
        # eps_now = 50.0*eps/(50+i)
        eps_now = eps
        lowest_error = float('inf')
        for j in range(n):
            for sign in [-1,1]:
                ws_test = ws.copy()
                ws_test[j] += sign*eps_now
                y_test = X_mat*ws_test
                rss_error = sum((Y_mat.A - y_test.A)**2)
                if rss_error<lowest_error:
                    lowest_error = rss_error
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i,:] = ws.T
    return return_mat







