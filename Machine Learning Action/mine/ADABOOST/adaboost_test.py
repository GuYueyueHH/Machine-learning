# -*- coding: utf-8 -*-

# @File    : adaboost_test.py
# @Date    : 2020-10-23
# @Author  : YUEYUE-x4
# @Demo    :  

#%%
print('adaboost test begin')
import ADABOOST
import imp
imp.reload(ADABOOST)
import numpy as np

data_mat,class_labels = ADABOOST.load_simple_dataset()
classifier_array = ADABOOST.adaboost_train(data_mat,class_labels,9)

classify = ADABOOST.adaboost_classify([[5,5],[0,0]],classifier_array)
# print(classify)

print('adaboost test end')

#%%
print('adaboost hoese colic test begin')
import ADABOOST
import imp
imp.reload(ADABOOST)
import numpy as np

def load_dataset(file_name):
    num_features = len(open(file_name).readline().split('\t'))
    data_mat = []
    class_labels = []
    f = open(file_name)
    for line in f.readlines():
        line_array = []
        cur_line = line.strip('\n').split('\t')
        for i in range(num_features-1):
            line_array.append(float(cur_line[i]))
        data_mat.append(line_array)
        class_labels.append(float(cur_line[-1]))
    return data_mat,class_labels

data_mat,class_labels = load_dataset('horseColicTraining2.txt')
classifier_array = ADABOOST.adaboost_train(data_mat,class_labels,100)

test_data_mat,test_class_labels = load_dataset('horseColicTest2.txt')
classify_labels = ADABOOST.adaboost_classify(test_data_mat,classifier_array)
error_array = np.mat(np.ones([len(classify_labels),1]))
error_num = error_array[classify_labels != np.mat(test_class_labels).T].sum()
print('test error rate:',1.0*error_num/len(classify_labels))

print('adaboost hoese colic test begin')

#%%
print('adaboost plot roc test begin')
import ADABOOST
import imp
imp.reload(ADABOOST)
import numpy as np

def load_dataset(file_name):
    num_features = len(open(file_name).readline().split('\t'))
    data_mat = []
    class_labels = []
    f = open(file_name)
    for line in f.readlines():
        line_array = []
        cur_line = line.strip('\n').split('\t')
        for i in range(num_features-1):
            line_array.append(float(cur_line[i]))
        data_mat.append(line_array)
        class_labels.append(float(cur_line[-1]))
    return data_mat,class_labels

data_mat,class_labels = load_dataset('horseColicTraining2.txt')
classifier_array,agg_class_est = ADABOOST.adaboost_train(data_mat,class_labels,50,output=1)
ADABOOST.plotROC(agg_class_est.T,class_labels)

print('adaboost plot roc test begin')

