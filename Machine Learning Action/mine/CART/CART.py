# -*- coding: utf-8 -*-

# @File    : CART.py
# @Date    : 2020-10-25
# @Author  : YUEYUE-x4
# @Demo    :

import numpy as np

# class tree_node():
#     def __init__(self,feature,value,right,left):
#

def load_dataset(filename):
    data_mat = []
    f = open(filename)
    for line in f.readlines():
        cur_line = line.strip().split('\t')
        features_labels_line = list(map(float,cur_line))
        data_mat.append(features_labels_line)
    f.close()
    return np.mat(data_mat)

def bin_split_dataset(dataset,feature,value):
    mat0 = dataset[ np.nonzero(dataset[:,feature]>value)[0],: ]
    mat1 = dataset[ np.nonzero(dataset[:,feature]<=value)[0],: ]
    return mat0,mat1

def reg_leaf(dataset):
    return np.mean(dataset[:,-1])

def reg_error(dataset):
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]

# 模型树：分段线性函数
def linear_solver(dataset):
    m,n = np.shape(dataset)
    X = np.mat(np.ones([m,n]))
    Y = np.mat(np.ones([m,1]))
    X[:,1:n] = dataset[:,0:n-1]
    Y = dataset[:,-1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse!')
    ws = xTx.I * (X.T*Y)
    return ws,X,Y

def model_leaf(dataset):
    ws,X,Y = linear_solver(dataset)
    return ws

def model_error(dataset):
    ws,X,Y = linear_solver(dataset)
    y_fit = X*ws
    return np.sum(np.power(Y-y_fit,2))

def choose_best_split(dataset, leaf_type=reg_leaf, error_type=reg_error, ops=(1,4),):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(dataset[:,-1].T.tolist()[0])) == 1:
        return None,leaf_type(dataset)
    m,n = np.shape(dataset)
    s = error_type(dataset)
    best_s = float('inf')
    best_index = 0
    best_value =0
    for feature_index in range(n-1):
        # print('dataset[:,feature_index].T.A',dataset[:,feature_index].T.A)
        for split_value in set(dataset[:,feature_index].T.A[0]):
            mat0,mat1 = bin_split_dataset(dataset,feature_index,split_value)
            if (np.shape(mat0)[0]<tol_n) or (np.shape(mat1)[0]<tol_n):
                continue
            new_s = error_type(mat0) + error_type(mat1)
            if new_s<best_s:
                best_index = feature_index
                best_value = split_value
                best_s = new_s
    if (s-best_s)<tol_s:
        return None, leaf_type(dataset)
    mat0,mat1 = bin_split_dataset(dataset, best_index, best_value)
    if (np.shape(mat0)[0]<tol_n) or (np.shape(mat1)[0]<tol_n):
        return None,leaf_type(dataset)
    return best_index, best_value

def create_tree(dataset, leaf_type=reg_leaf, error_type=reg_error, ops=(1,4)):
    feature,value = choose_best_split(dataset,leaf_type,error_type,ops)
    if feature==None:
        return value
    ret_tree = {}
    ret_tree['sp_ind'] = feature
    ret_tree['sp_val'] = value
    left_set,right_set = bin_split_dataset(dataset, feature, value)
    ret_tree['left'] = create_tree(left_set, leaf_type, error_type, ops)
    ret_tree['right'] = create_tree(right_set, leaf_type, error_type, ops)
    return ret_tree

# 回归树剪枝相关函数
def is_tree(obj):
    return (type(obj).__name__=='dict')

def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left'])/2.0

def prune(tree,test_data):
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    if is_tree(tree['left']) or is_tree(tree['right']):
        left_set,right_set = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], left_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right_set)
    if (not is_tree(tree['left'])) and (not is_tree(tree['right'])):
        left_set, right_set = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
        error_no_merge = np.sum( np.power(left_set[:,-1]-tree['left'],2)) + np.sum( np.power(right_set[:,-1]-tree['right'],2) )
        tree_mean = (tree['left']+tree['right'])/2.0
        error_merge = np.sum( np.power(test_data[:,-1]-tree_mean,2) )
        if error_merge<error_no_merge:
            # print('merging!')
            return tree_mean
        else:
            return tree
    else:
        return tree

# 预测
def reg_tree_eval(model,in_mat):
    return float(model)

def model_tree_eval(model,in_mat):
    n = np.shape(in_mat)[1]
    X = np.mat(np.ones([1,n+1]))
    X[:,1:n+1] = in_mat
    return float(X*model)

def tree_forecast(tree,in_mat,model_eval=reg_tree_eval):
    if not is_tree(tree):
        return model_eval(tree,in_mat)
    if in_mat[tree['sp_ind']]>tree['sp_val']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'],in_mat,model_eval)
        else:
            return model_eval(tree['left'],in_mat)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'],in_mat,model_eval)
        else:
            return model_eval(tree['right'],in_mat)

def create_forcast(tree,test_data,model_eval=reg_tree_eval):
    m = len(test_data)
    y_fit = np.mat(np.zeros([m,1]))
    for i in range(m):
        y_fit[i] = tree_forecast(tree,test_data[i],model_eval)
    return y_fit










