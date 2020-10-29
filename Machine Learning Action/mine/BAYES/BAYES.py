# -*- coding: utf-8 -*-

# @File    : BAYES.py
# @Date    : 2020-10-19
# @Author  : YUEYUE-x4
# @Demo    :  
import numpy as np
import math

# 创建句组成的列表
def create_dataset():
    posting_list = [['my','dog','has','flea','problems','help','please'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','steak','how','to','stop','him'],
                    ['quit','buying','worthless','dog','food','stupid']]
    class_vec = [0,1,0,1,0,1]
    return posting_list,class_vec

# 转换为没有重复的单词的列表
def create_vocabulary_list(dataset):
    vocabulary_set = set([])
    for line in dataset:
        vocabulary_set |= set(line)
    return list(vocabulary_set)

# 将输入的单词向量根据词汇表转换为0、1向量
def words_to_01vec(vocabulary,word_vec):
    word_01vec = [0]*len(vocabulary)
    for i in range(len(vocabulary)):
        if vocabulary[i] in word_vec:
            # 简单模型
            # word_01vec[i] = 1
            # 词袋模型
            word_01vec[i] += 1
    return word_01vec

# 计算两个分类占总分类的个数，计算每个分类中各个单词出现的频率
def train_NB0(train_matrix,train_class_vec):
    line_nums = len(train_matrix)
    p1 = sum(train_class_vec)/float(line_nums)
    # c0_sum = [0]*len(train_matrix[0])
    # c1_sum = [0]*len(train_matrix[0])
    c0_sum = np.ones(len(train_matrix[0]))
    c1_sum = np.ones(len(train_matrix[0]))
    for i in range(line_nums):
        if train_class_vec[i]:
            c1_sum += train_matrix[i]
        else:
            c0_sum += train_matrix[i]
    # vec_p0 = c0_sum/sum(c0_sum)
    # vec_p1 = c1_sum/sum(c1_sum)
    vec_p0 = np.log(1.0*c0_sum / (sum(c0_sum)-len(train_matrix[0])+2))
    vec_p1 = np.log(1.0*c1_sum / (sum(c1_sum)-len(train_matrix[0])+2))
    return vec_p0,vec_p1,p1

# 分类算法
def classify(test_vec,vec_p0,vec_p1,p1):
    p0_fit = sum(test_vec*vec_p0) + math.log(1-p1)
    p1_fit = sum(test_vec*vec_p1) + math.log(p1)
    if p1_fit > p0_fit:
        return 1
    else:
        return 0













