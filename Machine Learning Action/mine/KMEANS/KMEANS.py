# -*- coding: utf-8 -*-

# @File    : KMEANS.py
# @Date    : 2020-10-26
# @Author  : YUEYUE-x4
# @Demo    :  


import numpy as np

# 从文件载入数据
def load_dataset(filename):
    datamat = []
    f = open(filename)
    for line in f.readlines():
        cur_line = line.strip().split('\t')
        filter_line = map(float, cur_line)
        datamat.append(list(filter_line))
    f.close()
    return np.mat(datamat)

# 计算两组数据的欧式距离
def dist_eclud(veca,vecb):
    return np.sqrt( np.sum(np.power(veca-vecb,2)) )

# 随机k个质心
def rand_cent(dataset,k):
    n = np.shape(dataset)[1]
    cent_roids = np.mat( np.zeros([k,n]) )
    for i in range(n):
        min_col = np.min(dataset[:,i])
        range_col = float( np.max(dataset[:,i])-min_col )
        cent_roids[:,i] = min_col + range_col*np.random.rand(k,1)
    return cent_roids

def kmeans(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = np.shape(dataset)[0]
    cluster_assment = np.mat(np.zeros([m,2]))
    cent_roids = create_cent(dataset,k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = float('inf')
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(cent_roids[j,:],dataset[i,:])
                if dist_ji<min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i,0] != min_index:
                cluster_changed = True
            cluster_assment[i,:] = [min_index, min_dist**2]
        for cent in range(k):
            pts_in_clust = dataset[ np.nonzero(cluster_assment[:,0].A == cent)[0] ]
            cent_roids[cent,:] = np.mean(pts_in_clust, axis=0)
        # print('************',cent_roids)
    return cent_roids,cluster_assment

def bi_kmeans(dataset, k, dist_meas=dist_eclud):
    m = np.shape(dataset)[0]
    cluster_assment = np.mat(np.zeros([m,2]))
    cent_roid0 = np.mean(dataset, axis=0).tolist()[0]
    cent_list = [cent_roid0]
    for i in range(m):
        cluster_assment[i,1] = dist_meas(np.mat(cent_roid0),dataset[i,:])**2
    while len(cent_list)<k:
        lowest_sse = float('inf')
        for i in range(len(cent_list)):
            pts_incurr_cluster = dataset[ np.nonzero(cluster_assment[:,0]==i)[0],: ]
            cent_roid_mat,split_cluster_ass = kmeans(pts_incurr_cluster,2,dist_meas)
            sse_split = np.sum(split_cluster_ass[:,1])
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:,0].A!=i)[0],1])
            if (sse_split+sse_not_split)<lowest_sse:
                best_cent_tosplit = i
                best_new_cents = cent_roid_mat
                best_cluster_ass = split_cluster_ass.copy()
                lowest_sse = sse_split+sse_not_split
        best_cluster_ass[np.nonzero(best_cluster_ass[:,0].A == 1)[0],0] = len(cent_list)
        best_cluster_ass[np.nonzero(best_cluster_ass[:,0].A == 0)[0],0] = best_cent_tosplit
        cent_list[best_cent_tosplit] = best_new_cents[0,:].tolist()[0]
        cent_list.append(best_new_cents[1,:].tolist()[0])
        cluster_assment[np.nonzero(cluster_assment[:,0].A == best_cent_tosplit)[0],:] = best_cluster_ass
    return np.mat(cent_list),cluster_assment

# 球面距离
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * \
                      np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = bi_kmeans(datMat, numClust, dist_meas=distSLC)
    fig = plt.figure(1)
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()





