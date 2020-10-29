# -*- coding: utf-8 -*-

# @File    : SVM_simple.py
# @Date    : 2020-10-22
# @Author  : YUEYUE-x4
# @Demo    :  
import random
import numpy as np

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

def select_j_rand(i,m):
    j = i
    while j==i:
        j = int(random.uniform(0,m))
    return j

def clip_alpha(aj,H,L):
    if aj>H:
        aj = H
    if aj<L:
        aj = L
    return aj

def smo_simple(data,labels,C,toler,max_iters):
    data_matrix = np.mat(data)
    label_matrix = np.mat(labels).T
    # print(data_matrix)
    # print(label_matrix)
    b = 0
    m,n = np.shape(data_matrix)
    alphas = np.mat(np.zeros([m,1]))
    # print(alphas)
    iter = 0
    while iter<max_iters:
        alpha_pairs_changed = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,label_matrix).T * (data_matrix*data_matrix[i,:].T)) + b
            Ei = fXi - float(label_matrix[i])
            if ((label_matrix[i]*Ei < -toler) and (alphas[i] < C)) or ((label_matrix[i]*Ei > toler) and (alphas[i] > 0)):
                j = select_j_rand(i,m)
                fXj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_matrix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue
                eta = 2.0*data_matrix[i,:]*data_matrix[j,:].T - data_matrix[i,:]*data_matrix[i,:].T-data_matrix[j,:]*data_matrix[j,:].T
                if eta>= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= label_matrix[j]*(Ei-Ej)/eta
                alphas[j] = clip_alpha(alphas[j],H,L)
                if abs(alphas[j]-alpha_j_old)<0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += label_matrix[j]*label_matrix[i]*(alpha_j_old-alphas[j])
                b1 = b- Ei - label_matrix[i]*(alphas[i]-alpha_i_old)*data_matrix[i,:]*data_matrix[i,:].T - label_matrix[j]*(alphas[j]-alpha_j_old)*data_matrix[i,:]*data_matrix[j,:].T
                b2 = b- Ej - label_matrix[i]*(alphas[i]-alpha_i_old)*data_matrix[i,:]*data_matrix[j,:].T - label_matrix[j]*(alphas[j]-alpha_j_old)*data_matrix[j,:]*data_matrix[j,:].T
                if 0<alphas[i] and alphas[i]<C:
                    b = b1
                elif 0<alphas[j] and alphas[j]<C:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alpha_pairs_changed += 1
                print('iter:%d i:%d,pairs changed %d'%(iter,i,alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number:%d'%iter)
    return b,alphas