# -*- coding: utf-8 -*-

# @File    : bayes_test.py
# @Date    : 2020-10-19
# @Author  : YUEYUE-x4
# @Demo    :  

#%%
print('bayes_test begin')
import BAYES
import imp
imp.reload(BAYES)
posting_list,class_vec = BAYES.create_dataset()
vocabulary_list = BAYES.create_vocabulary_list(posting_list)
training_matrix = []
for line in posting_list:
    training_matrix.append(BAYES.words_to_01vec(vocabulary_list,line))
vec_p0,vec_p1,p1 = BAYES.train_NB0(training_matrix,class_vec)

# test_line = ['love','my','dalmation']
test_line = ['stupid','my','dalmation']
test_vec = BAYES.words_to_01vec(vocabulary_list,test_line)
classification = BAYES.classify(test_vec,vec_p0,vec_p1,p1)
print(classification)
print('bayes_test end')


#%%
print('bayes_test begin')
import BAYES
import imp
imp.reload(BAYES)
posting_list,class_vec = BAYES.create_dataset()
vocabulary_list = BAYES.create_vocabulary_list(posting_list)
training_matrix = []
for line in posting_list:
    training_matrix.append(BAYES.words_to_01vec(vocabulary_list,line))
vec_p0,vec_p1,p1 = BAYES.train_NB0(training_matrix,class_vec)

# test_line = ['love','my','dalmation']
test_line = ['stupid','my','dalmation']
test_vec = BAYES.words_to_01vec(vocabulary_list,test_line)
classification = BAYES.classify(test_vec,vec_p0,vec_p1,p1)
print(classification)
print('bayes_test end')



