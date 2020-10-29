# -*- coding: utf-8 -*-

# @File    : trees_test.py
# @Date    : 2020-10-15
# @Author  : YUEYUEHH
# @Demo    :

import TREES
#%%
print('TREE function test begin')
import TREES
import imp
imp.reload(TREES)
# TREES.create_dataset()生成dataset
dataset,labels = TREES.create_dataset()
print(dataset,labels)
# print(len(dataset))
shannon_ent = TREES.cal_shannon_ent(dataset)
# print(shannon_ent)
# selected_dataset = TREES.select_dataset(dataset,0,1)
# print(selected_dataset)
best_feature = TREES.choose_feature_tosplit(dataset)
# print('best_feature:',best_feature)
# print('TREE function test end')
tree = TREES.create_tree(dataset,labels)
print(tree,dataset,labels)
TREES.createPlot(tree,'tree1.png')
# TREES.pickle_tree(tree,'tree1.pickle')
# import pickle
# f = open('tree1.pickle','rb')
# tree1 = pickle.load(f)
# f.close()
# print(tree1)
test_vec = [1,0,0]
tree_class = TREES.classify_without_tree(dataset,labels,test_vec)
print(tree_class)

#%% 隐形眼睛
print('隐形眼睛 begin')
import TREES
import imp
imp.reload(TREES)
f = open(r'lenses.txt')
lense_data = f.readlines()
length_lense_data = len(lense_data)
for i in range(length_lense_data):
    lense_data[i] = lense_data[i].replace('\n','').split('\t')
    # print(len(lense_data[i]))
lense_labels = ['age','prescript','astigmatic','tearRate']
lense_tree = TREES.create_tree(lense_data,lense_labels)
TREES.createPlot(lense_tree,'lense_tree.png')
print(lense_tree)
f.close()

print('隐形眼睛 end')