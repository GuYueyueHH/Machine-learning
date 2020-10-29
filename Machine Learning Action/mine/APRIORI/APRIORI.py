# -*- coding: utf-8 -*-

# @File    : APRIORI.py
# @Date    : 2020-10-26
# @Author  : YUEYUE-x4
# @Demo    :  

import numpy as np

def create_dataset():
    # datamat = [[1,3,4],
    #            [2,3,5],
    #            [1,2,3,5],
    #            [2,5]]
    datamat = [[1,3,4],
               [2,3,5],
               [1,2,3,5],
               [2,5],
               [1],
               [1,2,3,4],
               [5,3,1]]
    return datamat

def create_c1(dataset):
    c1 = []
    for row in dataset:
        c1.extend(row)
    c1.sort()
    c1 = set(c1)
    # return [frozenset([item]) for item in c1]         # frozenset, 创建不可变集合 可以作为dict的key
    return [frozenset([item]) for item in c1]

def scan_d(d, ck, min_support):
    ss_cnt = {}
    for tid in d:
        for can in ck:
            if can.issubset(tid):
                ss_cnt[can] = ss_cnt.get(can, 0) + 1
    num_items = float(len(d))
    ret_list =[]
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key]/num_items
        if support >= min_support:
            ret_list.insert(0,key)
        support_data[key] = support
    return ret_list,support_data

def apriori_gen(lk,k):
    ck = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i+1,len_lk):
            l1 = list(lk[i])[:k-2]
            l2 = list(lk[j])[:k-2]
            l1.sort()
            l2.sort()
            if l1==l2:
                ck.append(lk[i] | lk[j])
    return ck

def apriori(dataset,min_support=0.5):
    c1 = create_c1(dataset)
    d = [set(item) for item in dataset]
    l1,support_data = scan_d(d,c1,min_support)
    l = [l1]
    k = 2
    while len(l[k-2])>0:
        ck = apriori_gen(l[k-2],k)
        lk,sup_k = scan_d(d,ck,min_support)
        if lk:
            support_data.update(sup_k)
            l.append(lk)
            k += 1
        else:
            break
    return l,support_data

def generater_rules(l, support_data, min_conf=0.7):  #supportData is a dict coming from scanD
    big_rule_list = []
    for i in range(1, len(l)):#only get the sets with two or more items
        for freq_set in l[i]:
            H = [frozenset([item]) for item in freq_set]
            rules_from_conseq(freq_set, H, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, H, support_data, big_rule_list, min_conf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = support_data[freq_set]/support_data[freq_set-conseq] #calc confidence
        if conf >= min_conf:
            # print (freq_set-conseq,'-->',conseq,'conf:',conf)
            big_rule_list.append((freq_set-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rules_from_conseq(freq_set, H, support_data, big_rule_list, min_conf=0.7):
    m = len(H[0])
    if m==1 and len(freq_set)>1:
        calc_conf(freq_set, H, support_data, big_rule_list, min_conf)
    while (len(freq_set) > (m + 1)):
        Hmp = apriori_gen(H, m + 1)
        Hmp = calc_conf(freq_set, Hmp, support_data, big_rule_list, min_conf)
        if not Hmp:
            break
        m = len(Hmp[0])
