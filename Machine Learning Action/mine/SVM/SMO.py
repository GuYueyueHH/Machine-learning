# -*- coding: utf-8 -*-

# @File    : SMO.py
# @Date    : 2020-10-22
# @Author  : YUEYUE-x4
# @Demo    :  
import numpy as np
import random

def load_dataset(filename):
    data_mat = []
    label_mat = []
    f = open(filename)
    for line in f.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]),float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    # print(label_mat)
    return data_mat,label_mat

class opt_struct:
    def __init__(self,data_m,labels,C,toker,k_tup):
        self.X = data_m
        self.labels_matrix = labels
        self.C = C
        self.toker = toker
        self.m = np.shape(data_m)[0]
        self.n = np.shape(data_m)[1]
        self.alphas = np.mat(np.zeros([self.m,1]))
        self.b = 0
        self.w = np.zeros([self.n, 1])
        self.e_cache = np.mat(np.zeros([self.m,2]))
        self.K = np.mat(np.zeros([self.m,self.m]))
        for i in range(self.m):
            self.K[:,i] = kernel_trans(self.X,self.X[i,:],k_tup)

def cal_ek(os,k):
    # fXk = float(np.multiply(os.alphas,os.labels_matrix).T * (os.X*os.X[k,:].T)) + os.b
    fXk = float(np.multiply(os.alphas,os.labels_matrix).T * os.K[:,k]) + os.b
    Ek = fXk - float(os.labels_matrix[k])
    return Ek

def select_j_rand(i,m):
    j = i
    while j==i:
        j = int(random.uniform(0,m))
    return j

def select_j(i,os,Ei):
    maxK = -1
    max_deltaE = 0
    Ej = 0
    os.e_cache[i] = [1,Ei]
    valid_ecache_list = np.nonzero(os.e_cache[:,0].A)[0]
    if len(valid_ecache_list)>1:
        for k in valid_ecache_list:
            if k==i:
                continue
            Ek = cal_ek(os,k)
            deltaE = abs(Ei-Ek)
            if deltaE>max_deltaE:
                maxK = k
                max_deltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = select_j_rand(i,os.m)
        Ej = cal_ek(os,j)
        return j,Ej

def update_Ek(os,k):
    Ek = cal_ek(os,k)
    os.e_cache[k] = [1,Ek]

def clip_alpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj

def inner_loop(i,os):
    Ei = cal_ek(os,i)
    if ((os.labels_matrix[i]*Ei < -os.toker) and (os.alphas[i] < os.C)) or ((os.labels_matrix[i]*Ei > os.toker) and (os.alphas[i] > 0)):
        j,Ej = select_j(i,os,Ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.labels_matrix[i] != os.labels_matrix[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L==H:
            # print('L==H')
            return 0
        # eta = 2.0*os.X[i,:]*os.X[j,:].T - os.X[i,:]*os.X[i,:].T - os.X[j,:]*os.X[j,:].T
        eta = 2.0*os.K[i,j] - os.K[i,i] - os.K[j,j]
        if eta >= 0:
            # print('eta >= 0')
            return 0
        os.alphas[j] -= os.labels_matrix[j]*(Ei - Ej)/eta
        os.alphas[j] = clip_alpha(os.alphas[j],H,L)
        update_Ek(os,j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            # print('j not moving enough')
            return 0
        os.alphas[i] += os.labels_matrix[j]*os.labels_matrix[i]*(alpha_j_old-os.alphas[j])
        update_Ek(os,i)
        # b1 = os.b - Ei - os.labels_matrix[i]*(os.alphas[i]-alpha_i_old)*os.X[i,:]*os.X[i,:].T - os.labels_matrix[j]*(os.alphas[j]-alpha_j_old)*os.X[i,:]*os.X[j,:].T
        # b2 = os.b - Ej - os.labels_matrix[i]*(os.alphas[i]-alpha_i_old)*os.X[i,:]*os.X[j,:].T - os.labels_matrix[j]*(os.alphas[j]-alpha_j_old)*os.X[j,:]*os.X[j,:].T
        b1 = os.b - Ei - os.labels_matrix[i]*(os.alphas[i]-alpha_i_old)*os.K[i,i] - os.labels_matrix[j]*(os.alphas[j]-alpha_j_old)*os.K[i,j]
        b2 = os.b - Ej - os.labels_matrix[i]*(os.alphas[i]-alpha_i_old)*os.K[i,j] - os.labels_matrix[j]*(os.alphas[j]-alpha_j_old)*os.K[j,j]
        if (0<os.alphas[i]) and (os.alphas[i]<os.C):
            os.b = b1
        elif (0<os.alphas[j]) and (os.alphas[j]<os.C):
            os.b = b2
        else:
            os.b = (b1+b2)/2.0
        return 1
    else:
        return 0

def smo(data,labels,C,toker,max_iter,k_tup=('lin',0)):
    os = opt_struct(np.mat(data),np.mat(labels).T,C,toker,k_tup)
    iter = 0
    entire_set =True
    alpha_pairs_changed = 0
    while (iter<max_iter) and ((alpha_pairs_changed>0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += inner_loop(i,os)
            # print('fullset,iter:%d i:%d,pairs changed %d'%(iter,i,alpha_pairs_changed))
            iter += 1
        else:
            non_bound_list = np.nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]
            for i in non_bound_list:
                alpha_pairs_changed += inner_loop(i,os)
            # print('non bound,iter:%d i:%d,pairs changed %d'%(iter,i,alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed==0:
            entire_set = True
        # print('iteration number:%d'%iter)
    # while (iter<max_iter):
    #     alpha_pairs_changed = 0
    #     for i in range(os.m):
    #         alpha_pairs_changed += inner_loop(i,os)
    #     # print('fullset,iter:%d i:%d,pairs changed %d'%(iter,i,alpha_pairs_changed))
    #     if alpha_pairs_changed == 0:
    #         iter += 1
    #     print('iteration number:%d'%iter)

    if k_tup[0]=='lin':
        for i in range(os.m):
            os.w += np.multiply(os.alphas[i]*os.labels_matrix[i],os.X[i,:].T)
    elif k_tup[0]=='rbf':
        os.w = np.multiply(os.labels_matrix, os.alphas)
    return os.w,os.b,os.alphas

def SVM_classify(data_mat_in,w,b,data_mat_train=np.mat(np.zeros([10,10])),k_tup=('lin',0)):
    if k_tup[0]=='lin':
        SVM_fit = data_mat_in*w + b
    elif k_tup[0] == 'rbf':
        kernel_eval = kernel_trans(np.mat(data_mat_train),np.mat(data_mat_in),k_tup)
        SVM_fit = kernel_eval.T*w + b
    if SVM_fit > 0:
        return 1
    else:
        return -1


# 径向基函数内积
def kernel_trans(X,A,k_tup):
    m,n = np.shape(X)
    K = np.mat(np.zeros([m,1]))
    if k_tup[0] == 'lin':
        K = X*A.T
    elif k_tup[0] == 'rbf':
        for i in range(m):
            delta_row = X[i,:] - A
            K[i] = delta_row*delta_row.T
        K = np.exp(K / (-1*k_tup[1]**2))
    return K









