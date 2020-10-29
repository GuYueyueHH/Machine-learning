# -*- coding: utf-8 -*-

# @File    : kNN_test.py
# @Date    : 2020-10-01
# @Author  : YUEYUE-x4
# @Demo    :


# %%
print('classify test begin')
import numpy as np
import KNN
from tqdm import tqdm
# trange
def classifyWithClassify0():
    # 检验 KNN.py 中的 classify() 和 classify0() 是否相同
    # 相同：count == 0
    count = 0
    countSame = 0
    k = 3
    # dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    dataSet = np.array([[1.0, 1.1, 1.1], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    for i in tqdm(np.arange(-1,1,0.01)):
        for j in np.arange(-1, 1, 0.01):
            for z in np.arange(-1, 1, 0.01):
                inX = [i,j,z]
                classifyResult = KNN.classify(inX,dataSet,labels,k)
                classify0Result = KNN.classify0(inX, dataSet, labels, k)
                if classifyResult != classify0Result:
                    count += 1
                else:
                    countSame += 1
    return count,countSame
# count,countSame = classifyWithClassify0()
# print(count,countSame)


def classifyWithClassifyCupy():
    # 检验 KNN.py 中的 classify() 和 classifyCupy() time
    import cupy as cp
    import time
    k = 3
    labels = ['A', 'A', 'B', 'B']

    time0 = time.time()
    dataSet = np.array([[1.0, 1.1, 1.1], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0.1, 0.1]])
    for i in tqdm(np.arange(-1, 1, 0.01)):
        for j in np.arange(-1, 1, 0.01):
            for z in np.arange(-1, 1, 0.01):
                inX = [i, j, z]
                KNN.classify(inX, dataSet, labels, k)
        print("i from numpy:",i)
    time0 = time.time() - time0

    timeCupy = time.time()
    # dataSet = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    dataSet = cp.array([[1.0, 1.1, 1.1], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0.1, 0.1]])
    for i in cp.arange(-1, 1, 0.01):
        for j in cp.arange(-1, 1, 0.01):
            for z in cp.arange(-1, 1, 0.01):
                inX = cp.array([i, j, z])
                KNN.classifyCupy(inX, dataSet, labels, k)
        print("i from numpy:", i)
    timeCupy = time.time() - timeCupy

    return time0,timeCupy
# time0,timeCupy = classifyWithClassifyCupy()
# print(time0,timeCupy)
print('classify test end')
# %%
print('datingTestSet.txt 测试 begin')
# datingTestSet.txt 测试
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

os.chdir('D:\OneDrive - zju.edu.cn\Documents\Machine learning\Machine learning action\mine\KNN')
import KNN
import importlib
importlib.reload(KNN)

def dating_txt2matrix(datingFile):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    f = open(datingFile,'r',encoding='utf8')
    data = f.readlines()
    datingMatrix = np.zeros([len(data),3])
    datingLabel = []
    index = 0
    for line in data:
        # print(line)
        line = line.strip().split('	')
        datingMatrix[index,:] = line[0:3]
        if line[3].isdigit():
            datingLabel.append(line[3])
        else:
            datingLabel.append(int(love_dictionary.get(line[3],0)))
        index += 1

    def norm():
        for i in range(3):
            datingMatrix[:,i] = ( datingMatrix[:,i]-min(datingMatrix[:,i]) ) / ( max(datingMatrix[:,i])-min(datingMatrix[:,i]) )
    norm()

    return datingMatrix,datingLabel

def scatterPlot(Matrix,labels):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    new_love_dict = {v:k for k,v in love_dictionary.items()}
    uniqueLabels = np.unique(labels)
    legends = [new_love_dict[label] for label in uniqueLabels]
    marker = ['x','*','o','1']
    index = 0
    fig = plt.figure(figsize=(6,6))
    for label in uniqueLabels:
        labelIndexes = [i for i,x in enumerate(labels) if x==label]
        plt.scatter(Matrix[labelIndexes,0],Matrix[labelIndexes,1],marker=marker[index])
        index += 1
    plt.xlabel('datingMatrix[:,0]')
    plt.ylabel('datingMatrix[:,1]')
    plt.title('datingMatrix')
    plt.legend(legends)
    plt.show()
# datingMatrix,datingLabel = dating_txt2matrix('datingTestSet.txt')
# scatterPlot(datingMatrix[:,0:2],datingLabel)

# datingTestSet.txt的一部分数据测试KNN效果
def dating_knn_test(trainning_ratio,k):
    dating_matrix, dating_label = dating_txt2matrix('datingTestSet.txt')
    matrix_length = dating_matrix.shape[0]
    sample_length = int(trainning_ratio*matrix_length)
    trainning_matrix = dating_matrix[0:sample_length,:]
    trainning_label = dating_label[0:sample_length]
    test_matrix = dating_matrix[sample_length:, :]
    test_label = dating_label[sample_length:]
    test_length = test_matrix.shape[0]
    true_length = 0
    for i in range(test_length):
        # print(i)
        label = KNN.classify(test_matrix[i,:],trainning_matrix,trainning_label,k)
        # print(label-test_label[i])
        if label-test_label[i]==0:
            true_length += 1
    return 1.0*true_length/test_length
# ratio = dating_knn_test(0.5,3)
# print('truth ratio: %.2f%%\nfalse ratio: %.2f%%'%(ratio*100,100-ratio*100))

# k变化时 查看正确率变化
# max:3
def k_dating_KNN():
    k_range = list(range(1,21))
    ratio_range = list(np.arange(0.1,1,0.05))
    accuracy = np.zeros([len(k_range),len(ratio_range)])
    i=0
    for k in k_range:
        j = 0
        for ratio in ratio_range:
            accuracy[i,j] = dating_knn_test(ratio,k)
            # print(i,j)
            j += 1
        # print(i)
        i += 1
    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig)
    k, ratio = np.meshgrid(k_range, ratio_range)
    accuracy = accuracy.T
    # print(k)
    print(k.shape,ratio.shape,accuracy.shape)
    p = ax.plot_surface(k, ratio, accuracy, rstride=1, cstride=1, cmap='rainbow')
    plt.xlabel('k range')
    plt.ylabel('ratio range')
    plt.title('accuracy of k and ratio')
    fig.colorbar(p)
    plt.show()
k_dating_KNN()
print('datingTestSet.txt 测试 end')
#%% 手写数字识别
print('手写数字识别 begin')
import numpy as np
import KNN

# 得到digits下testDigits和trainingDigits下的文件的文件名
import os
# print(os.getcwd())        # 查看当前目录
main_path = os.getcwd()
test_digits_path = os.path.join(main_path,'digits\\testDigits')
train_digits_path = os.path.join(main_path,'digits\\trainingDigits')
# print(digits_path)        # 查看拼接是否正确
test_files_name = os.listdir(test_digits_path)
train_files_name = os.listdir(train_digits_path)

# 得到训练集和测试测试集的matrix和label
def get_matrix_label(files_dir,files_list):
    files_num = len(files_list)
    matrixs = np.zeros([files_num,1024])
    labels = []
    for i in range(files_num):
        file = files_list[i]
        file_dir = os.path.join(files_dir,file)
        # print(file_dir)
        f = open(file_dir,'r')
        data = map(lambda x:int(x),list(f.read().replace('\n','')))
        # print(data)
        f.close()
        matrix = np.array(list(data))
        label = file[0]
        matrixs[i,:] = matrix
        labels.append(label)
    return matrixs,labels

train_matrixs,train_labels = get_matrix_label(train_digits_path,train_files_name)
test_matrixs,test_labels = get_matrix_label(test_digits_path,test_files_name)
k = 3
error_count = 0
for i in range(len(test_matrixs)):
    test_matrix = test_matrixs[i,:]
    test_label = test_labels[i]
    label_output = KNN.classify(test_matrix, train_matrixs, train_labels, k)
    if label_output != test_label:
        error_count += 1
    error_rate = error_count/len(test_matrixs)
    print('Round: %d / %d, label: %s - %s, error rate: %.2f%%'%(i, len(test_matrixs), test_label, label_output, error_rate*100))

print('错误率为：%.2f%%'%(error_rate*100))

print('手写数字识别 end')
































