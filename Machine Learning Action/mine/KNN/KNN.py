# -*- coding: utf-8 -*-

# @File    : KNN.py
# @Date    : 2020-10-01
# @Author  : YUEYUE-x4
# @Demo    : python 3.7

import numpy as np

def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    # print('form KNN classify:',diffMat)
    diffArray = (diffMat**2).sum(axis=1)**0.5
    diffSort = diffArray.argsort()
    classifyDict = {}
    for i in range(k):
        classifyDict[labels[diffSort[i]]] = classifyDict.get(labels[diffSort[i]],0) + 1
    return max(classifyDict,key=classifyDict.get)

def classifyCupy(inX,dataSet,labels,k):
    import cupy as cp
    dataSetSize = dataSet.shape[0]
    diffMat = cp.tile(inX,[dataSetSize,1])-dataSet
    # print('form KNN classify:',diffMat)
    diffArray = (diffMat**2).sum(axis=1)**0.5
    diffSort = diffArray.argsort()
    print('\n',diffSort,'\n',int(diffSort[0]))
    classifyDict = {}
    for i in range(k):
        classifyDict[labels[int(diffSort[i])]] = classifyDict.get(labels[int(diffSort[i])],0) + 1
    return max(classifyDict,key=classifyDict.get)

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # print(distances, labels, sep='\n')
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = max(classCount, key=classCount.get)
    return sortedClassCount

# inX = np.array([1, 0.05])
# k = 3
# dataSet = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
# labels = ['A','A','B','B']
# print(classify(inX,dataSet,labels,k))
