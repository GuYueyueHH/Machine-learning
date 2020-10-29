# -*- coding: utf-8 -*-

# @File    : TREES.py
# @Date    : 2020-10-15
# @Author  : YUEYUEHH
# @Demo    : 决策树

import numpy as np
import math

def create_dataset():
    dataset = [[1,1,1,'yes'],
               [1,0,1,'yes'],
               [1,1,0,'no'],
               [0,0,1,'no'],
               [0,1,1,'no']]
    labels = ['no surfacing','flippers','extra']
    return dataset,labels

# 计算香农熵
def cal_shannon_ent(dataset):
    data_nums = len(dataset)
    labels_count = {}
    for line in dataset:
        label = line[-1]
        labels_count[label] = labels_count.get(label,0) + 1
    shannon_ent = 0.0
    for key,value in labels_count.items():
        prob = 1.0*value/data_nums
        shannon_ent -= prob*math.log(prob,2)
    return shannon_ent

# 根据某一维的特定值选取数据集
def select_dataset(dataset, axis, value):
    selected_dataset = []
    for line in dataset:
        if line[axis] == value:
            selected_line_data = line[:axis]
            selected_line_data.extend(line[axis+1:])
            selected_dataset.append(selected_line_data)
    return selected_dataset

# 选择熵最大的维度去分割数据
def choose_feature_tosplit(dataset):
    shannon_ent_origin = cal_shannon_ent(dataset)
    # print(shannon_ent_origin)
    best_feature = -1
    features = len(dataset[0])-1
    lines_num = len(dataset)
    for feature in range(features):
        feature_values = set([line[feature] for line in dataset])
        shannon_ent = 0.0
        for feature_value in feature_values:
            selected_dataset = select_dataset(dataset, feature, feature_value)
            prob = 1.0*len(selected_dataset)/lines_num
            shannon_ent += prob*cal_shannon_ent(selected_dataset)
        # print(feature,shannon_ent)
        if shannon_ent < shannon_ent_origin:
            # 熵减小
            best_feature = feature
            shannon_ent_origin = shannon_ent
    return best_feature

# 根据训练集生成tree
def create_tree(dataset,labels):
    feature_labels = labels.copy()
    class_list = [line[-1] for line in dataset]
    features_num = len(dataset[0]) - 1
    # 如果都属于1个分类
    if len(set(class_list)) == 1:
        return class_list[0]
    # 如果只有1个特征，选取个数最多的那个分类
    if features_num==1 and len(set([line[0] for line in dataset]))==1:
        class_of_dataset_num = {}
        for class_of_dataset in class_list:
            class_of_dataset_num[class_of_dataset] = class_of_dataset_num.get(class_of_dataset,0) + 1
        return max(class_of_dataset_num,key=class_of_dataset_num.get)
    best_feature = choose_feature_tosplit(dataset)
    best_feature_label = feature_labels[best_feature]
    best_feature_values = set([line[best_feature] for line in dataset])
    tree = {best_feature_label: {}}
    del feature_labels[best_feature]
    for best_feature_class in best_feature_values:
        selected_dataset = select_dataset(dataset, best_feature, best_feature_class)
        tree[best_feature_label][best_feature_class] = create_tree(selected_dataset,feature_labels)
    return tree

# pickle存储tree
def pickle_tree(tree,tree_name):
    import pickle
    f = open(tree_name,'wb')
    pickle.dump(tree,f)
    f.close()

# 根据训练集生成tree，并输出测试集分类
def classify_without_tree(dataset,labels,test_vec):
    tree = create_tree(dataset, labels)
    # print('tree from classify_without_tree:',tree)
    feature_vec_dict = {key:value for key,value in zip(labels,test_vec)}
    # tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    while type(tree[list(tree.keys())[0]][feature_vec_dict[list(tree.keys())[0]]]).__name__ == 'dict':
        tree = tree[list(tree.keys())[0]][feature_vec_dict[list(tree.keys())[0]]]
    return list(tree.values())[0][feature_vec_dict[list(tree.keys())[0]]]

# 根据输入tree，输出测试集分类
def classify_with_tree(tree,labels,test_vec):
    print('tree from classify_with_tree:',tree)
    feature_vec_dict = {key:value for key,value in zip(labels,test_vec)}
    # tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    while type(tree[list(tree.keys())[0]][feature_vec_dict[list(tree.keys())[0]]]).__name__ == 'dict':
        tree = tree[list(tree.keys())[0]][feature_vec_dict[list(tree.keys())[0]]]
    return list(tree.values())[0][feature_vec_dict[list(tree.keys())[0]]]


import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0] + 0.02
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30,fontsize=10)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree,png_name):
    fig = plt.figure(1, facecolor='white',figsize=(8,6))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.savefig(png_name, dpi=1200)
    plt.show()





