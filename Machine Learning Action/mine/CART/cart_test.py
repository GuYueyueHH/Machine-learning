# -*- coding: utf-8 -*-

# @File    : cart_test.py
# @Date    : 2020-10-25
# @Author  : YUEYUE-x4
# @Demo    :  
print('cart test begin!')
#%%
print('cart test begin!')
import CART
import imp
imp.reload(CART)
import numpy as np

test_flag = 3

if test_flag == 0:
    test_mat = np.mat(np.eye(4))
    mat0,mat1 = CART.bin_split_dataset(test_mat,2,0.5)
    # print('mat0:',mat0)
    # print('mat1:',mat1)

elif test_flag == 1:
    # my_mat = np.mat(CART.load_dataset('ex00.txt'))
    my_mat = np.mat(CART.load_dataset('ex0.txt'))
    cart_tree = CART.create_tree(my_mat)
    print(cart_tree)
    # output = {'sp_ind': 1, 'sp_val': 0.400158, 'left': {'sp_ind': 1, 'sp_val': 0.797583, 'left': 3.9871632, \
    #                                                                                     'right': {'sp_ind': 1, 'sp_val': 0.609483, 'left': 2.9775723414634148, 'right': 1.9842189268292687}}, \
    #                                             'right': {'sp_ind': 1, 'sp_val': 0.208197, 'left': 1.0131366551724137, 'right': -0.023838155555555553}}

elif test_flag == 2:
    # 后剪枝
    datamat_ex2 = CART.load_dataset('ex2.txt')
    tree_ex2 = CART.create_tree(datamat_ex2,ops=(0,1))
    print('tree ex2:',tree_ex2)
    datamat_ex2test = CART.load_dataset('ex2test.txt')
    tree_ex2test = CART.prune(tree_ex2, datamat_ex2test)
    print('tree ex2test:',tree_ex2test)

elif test_flag == 3:
    # 模型树test
    datamat_exp2 = CART.load_dataset('exp2.txt')
    tree_exp2_model = CART.create_tree(datamat_exp2, CART.model_leaf, CART.model_error, ops=(1,10))
    print('tree exp2 model:',tree_exp2_model)
print('cart test end')
#%%
print('cart fit begin!')
import CART
import imp
imp.reload(CART)
import numpy as np

train_mat = CART.load_dataset('bikeSpeedVsIq_train.txt')
test_mat = CART.load_dataset('bikeSpeedVsIq_test.txt')
tree_train = CART.create_tree(train_mat,ops=(1,20))
y_fit = CART.create_forcast(tree_train,test_mat[:,0])
corr = np.corrcoef(test_mat[:,1].T.tolist()[0],y_fit.T.tolist()[0])
print('corr:',corr)

# ws,X,Y = CART.linear_solver(train_mat)

print('cart fit begin!')
