# -*- coding: utf-8 -*-

# @File    : svm_test.py
# @Date    : 2020-10-22
# @Author  : YUEYUE-x4
# @Demo    :  

#%%
print('svm simple test begin')
import SVM_simple
import imp
imp.reload(SVM_simple)
data_mat,label_mat = SVM_simple.load_dataset('testSet.txt')
b,alphas = SVM_simple.smo_simple(data_mat,label_mat,0.6,0.001,40)
print(b)
print(alphas[alphas>0])

print('svm simple test end')


#%%
print('smo test begin')
import SMO
import numpy as np
import imp
imp.reload(SMO)
data_mat,label_mat = SMO.load_dataset('testSet.txt')
w,b,alphas = SMO.smo(data_mat,label_mat,0.6,0.001,40)
print('w:',w,'b:',b)
print('alphas[alphas>0]',alphas[alphas>0])

error_count = 0
for i in range(len(data_mat)):
    testing_data = SMO.SVM_classify(np.mat(data_mat[i]),w,b)
    # print('classification with class label:  %d -- %d'%(testing_data,label_mat[i]))
    if testing_data != label_mat[i]:
        error_count += 1
print('error rate:%.2f%%'%(100.0*error_count/len(data_mat)))
print('smo test end')

#%%
print('smo RBF test begin')
import SMO
import numpy as np
import imp
imp.reload(SMO)
data_mat,label_mat = SMO.load_dataset('testSetRBF.txt')
w,b,alphas = SMO.smo(data_mat,label_mat,200,0.0001,50000,('rbf',1.25))
# print('w:',w,'b:',b)
# print('alphas[alphas>0]',alphas[alphas>0])
error_count = 0
for i in range(len(data_mat)):
    svm_fit = SMO.SVM_classify(np.mat(data_mat[i]),w,b,data_mat_train=data_mat,k_tup=('rbf',1.25))
    # print('classification with class label:  %d -- %d'%(svm_fit,label_mat[i]))
    if svm_fit != label_mat[i]:
        error_count += 1
print('\n\ntestSetRBF error rate:%.2f%%'%(100.0*error_count/len(data_mat)))
data_mat2,label_mat2 = SMO.load_dataset('testSetRBF2.txt')
error_count = 0
for i in range(len(data_mat2)):
    svm_fit = SMO.SVM_classify(np.mat(data_mat2[i]),w,b,data_mat_train=data_mat,k_tup=('rbf',1.25))
    # print('classification with class label:  %d -- %d'%(svm_fit,label_mat2[i]))
    if svm_fit != label_mat2[i]:
        error_count += 1
print('testSetRBF2 error rate:%.2f%%'%(100.0*error_count/len(data_mat2)))

# data_mat2,label_mat2 = SMO.load_dataset('testSetRBF2.txt')
# min_error_rate = 1
# min_error_rate_k1 = 0
# min_error_rate_k2 = 0
# for k1 in np.arange(0.1,2.0,0.1):
#     w,b,alphas = SMO.smo(data_mat,label_mat,200,0.0001,10000,('rbf',k1))
#     for k2 in np.arange(0.1,1.5,0.05):
#         error_count = 0
#         for i in range(len(data_mat2)):
#             svm_fit = SMO.SVM_classify(np.mat(data_mat2[i]),w,b,data_mat_train=data_mat,k_tup=('rbf',k2))
#             # print('classification with class label:  %d -- %d'%(svm_fit,label_mat2[i]))
#             if svm_fit != label_mat2[i]:
#                 error_count += 1
#         if 1.0*error_count/len(data_mat2) < min_error_rate:
#             min_error_rate = error_count/len(data_mat2)
#             min_error_rate_k1 = k1
#             min_error_rate_k1 = k2
#         print('%.2f - %.2f : testSetRBF2 error rate:%.2f%%'%(k1,k2,100.0*error_count/len(data_mat2)))
# print("**********k1:%.2f  k2:%.2f  min error rate:%.2f%%"%(min_error_rate_k1,min_error_rate_k1,100*min_error_rate))
print('smo RBF test end')