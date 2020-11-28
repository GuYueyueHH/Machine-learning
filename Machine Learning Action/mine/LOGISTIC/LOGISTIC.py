import numpy as np

def load_dataset():
    data_mat = []
    label_mat = []
    f = open('datasets/testSet.txt')
    for line in f.readlines():
        line_array = line.strip().split()
        data_mat.append([1.0, float(line_array[0]), float(line_array[1])])
        label_mat.append(int(line_array[2]))
    return data_mat, label_mat

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

def grad_ascent(data_mat,label_mat):
    data_matrix = np.mat(data_mat)
    label_matrix = np.mat(label_mat).T
    m,n = data_matrix.shape
    print('m,n:',m,n)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n,1))
    # print('weights:',weights)
    for k in range(max_cycles):
        h = sigmoid(data_matrix*weights)
        error = label_matrix - h
        # print('shape of error:',error.shape)
        # print('shape:',alpha*data_matrix.T*error)
        weights = weights + alpha*(data_matrix.T*error)
    return weights

def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat,label_mat = load_dataset()
    data_array = np.array(data_mat)
    # weights = grad_ascent(data_mat,label_mat)
    if type(weights).__name__ == 'matrix':
        weights = weights.getA()
    m = data_array.shape[0]
    x_cord1 = []
    x_cord2 = []
    y_cord1 = []
    y_cord2 = []
    for i in range(m):
        if int(label_mat[i]):
            x_cord1.append(data_array[i,1])
            y_cord1.append(data_array[i,2])
        else:
            x_cord2.append(data_array[i,1])
            y_cord2.append(data_array[i,2])
    fig = plt.figure(1)
    plt.scatter(x_cord1,y_cord1,s=30,c='r',marker='s')
    plt.scatter(x_cord2,y_cord2,s=30,c='g',marker='*')
    x = np.arange(-4.0,4.0,0.05)
    y = (-weights[0]-weights[1]*x)/weights[2]
    plt.plot(x,y)
    plt.show()

def stoc_grad_ascent(data_mat,label_mat):
    data_mat = np.array(data_mat)
    m,n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones((1,n))
    # print('weights shape:',weights.shape)
    for i in range(m):
        h = sigmoid(sum(data_mat[i]*weights))
        error = label_mat[i] - h
        weights = weights + alpha*error*data_mat[i]
    return weights.transpose()   

import random
def stoc_grad_ascent1(data_mat,label_mat,iterations=150):
    data_mat = np.array(data_mat)
    m,n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones((1,n))
    weights_matrix = np.zeros((iterations*m,n))
    weights_i = 0
    for j in range(iterations):
        data_index = list(range(m))
        for i in data_index:
            # random_index = random.randint(0,m-1)
            alpha = 4/(1.0+i+j*m) + 0.001
            h = sigmoid(sum(data_mat[i]*weights))
            error = label_mat[i] - h
            weights = weights + alpha*error*data_mat[i]
            weights_matrix[weights_i,:] = weights
            weights_i += 1
    return weights.transpose(),weights_matrix
















