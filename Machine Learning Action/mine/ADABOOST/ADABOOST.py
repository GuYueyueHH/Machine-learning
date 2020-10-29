# -*- coding: utf-8 -*-

# @File    : ADABOOST.py
# @Date    : 2020-10-23
# @Author  : YUEYUE-x4
# @Demo    :  
import numpy as np

def load_simple_dataset():
    data_mat = np.mat([[1.0,2.1],
                       [2.0,1.0],
                       [1.3,1.0],
                       [1.0,1.0],
                       [2.0,1.0]])
    class_list = [1.0,1.0,-1.0,-1.0,1.0]
    return data_mat,class_list

def stump_classify(data_mat,dimen,thresh_val,thresh_ineq):
    ret_array = np.ones([np.shape(data_mat)[0],1])
    if thresh_ineq == 'lt':
        ret_array[data_mat[:,dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array

def build_stump(data_mat,class_labels,D):
    data_mat = np.mat(data_mat)
    label_mat = np.mat(class_labels).T
    m,n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros([m,1]))
    min_error = float('inf')
    for i in range(n):
        range_min = min(data_mat[:,i])
        range_max = max(data_mat[:,i])
        step_size = (range_max - range_min)/num_steps
        for j in range(int(num_steps)+1):
            for inequal in ['lt','gt']:
                thresh_val = range_min + step_size*j
                predict_vals = stump_classify(data_mat,i,thresh_val,inequal)
                error_array = np.mat(np.ones([m,1]))
                error_array[predict_vals == label_mat] = 0
                weighted_error = D.T*error_array
                # print('split:dim %d,thresh %.2f,thresh inequal:%s,the weighted error is %.3f'%(i,thresh_val,inequal,weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predict_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump,min_error,best_class_est

def adaboost_train(data_mat,class_labels,num_iters=40,output=0):
    weak_class_array = []
    m,n = np.shape(data_mat)
    D = np.mat(1.0*np.ones([m,1])/m)
    agg_class_est = np.mat(np.zeros([m,1]))
    for i in range(num_iters):
        best_stump,error,class_est = build_stump(data_mat,class_labels,D)
        # print("D:",D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        best_stump['alpha'] = alpha
        weak_class_array.append(best_stump)
        # print('class est:',class_est.T)
        expon = np.multiply(-1.0*alpha*np.mat(class_labels).T,class_est)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        agg_class_est += alpha*class_est
        # print('agg class est:',agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T,np.ones([m,1]))
        error_rate = agg_errors.sum()/m
        print('iter:%d total error:%.3f'%(i,error_rate))
        if error_rate==0.0:
            break
    if output == 0:
        return weak_class_array
    else:
        return weak_class_array,agg_class_est

def adaboost_classify(data_mat,classifier):
    data_mat = np.mat(data_mat)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros([m,1]))
    for i in range(len(classifier)):
        class_est = stump_classify(data_mat,classifier[i]['dim'],
                                   classifier[i]['thresh'],
                                   classifier[i]['ineq'])
        agg_class_est += classifier[i]['alpha']*class_est
    return np.sign(agg_class_est)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    # print('1',sortedIndicies)
    # print('2',sortedIndicies.tolist()[0])
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)






















