# -*- coding: utf-8 -*-

# @File    : fpgrowth_test.py
# @Date    : 2020-10-27
# @Author  : YUEYUE-x4
# @Demo    :  
print('FP growth test!')


#%%
print('FP growth test begin!')
import FPGROWTH
import imp
imp.reload(FPGROWTH)
import numpy as np

test_flag = 1

if test_flag == 0:
    root_node = FPGROWTH.tree_node('pyraid',9,None)
    root_node.children['eye'] = FPGROWTH.tree_node('eye',13,None)
    root_node.children['phoenix'] = FPGROWTH.tree_node('phoenix',3,None)
    root_node.disp()
elif test_flag == 1:
    simple_dataset = FPGROWTH.load_simple_dataset()
    # print(simple_dataset)
    init_set = FPGROWTH.create_initset(simple_dataset)
    # print(init_set)
    my_fptree,my_header_table = FPGROWTH.create_tree(init_set,3)
    print('my_header_table:',my_header_table)
    # my_fptree.disp()
    item_prefix_paths = FPGROWTH.find_prefix_path('r',my_header_table['r'][1])
    # print(item_prefix_paths)
    freq_items = []
    FPGROWTH.mine_tree(my_fptree, my_header_table, 3, set([]), freq_items)
    for item in freq_items:
        print('freq item:',item)
    print('nums:',len(freq_items))
print('FP growth test end!')

