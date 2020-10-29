# -*- coding: utf-8 -*-

# @File    : apriori_test.py
# @Date    : 2020-10-26
# @Author  : YUEYUE-x4
# @Demo    :  
print('This is apriori test!')

#%%
print('Apriori test begin!')
import APRIORI
import imp
imp.reload(APRIORI)
import numpy as np
test_flag =2
if test_flag == 0:
    dataset = APRIORI.create_dataset()
    c1 = APRIORI.create_c1(dataset)
    d = [set(item) for item in dataset]
    l1,support_data0 = APRIORI.scan_d(d,c1,0.5)
    print(l1)

elif test_flag == 1:
    dataset = APRIORI.create_dataset()
    l,support_data = APRIORI.apriori(dataset,0.5)
    print('l:',l)
    print('support_data:',support_data)

elif test_flag == 2:
    dataset = APRIORI.create_dataset()
    l, support_data = APRIORI.apriori(dataset, 0.5)
    rules = APRIORI.generater_rules(l,support_data,min_conf=0.1)
    for rule in rules:
        print('rules:',rule)
    print('rules length:', len(rules))

print('Apriori test end!')