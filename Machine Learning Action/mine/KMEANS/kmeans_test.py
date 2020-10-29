# -*- coding: utf-8 -*-

# @File    : kmeans_test.py
# @Date    : 2020-10-26
# @Author  : YUEYUE-x4
# @Demo    :  

print('k-means test begin')

#%%
print('k means test begin!')
import KMEANS
import imp
imp.reload(KMEANS)
import numpy as np

dataset = KMEANS.load_dataset('testSet.txt')
centers = KMEANS.rand_cent(dataset,2)
# distance = KMEANS.dist_eclud(dataset[0],dataset[1])
cent_roids,cluster_assment = KMEANS.kmeans(dataset,4)
print(cent_roids)
cent_roids,cluster_assment = KMEANS.bi_kmeans(dataset,4)
print(cent_roids)
print('k means test end!')

#%%
print('yahoo api test begin!')
import KMEANS
import imp
imp.reload(KMEANS)
import numpy as np

# # 不能用
# import urllib
# # import urllib.request
# import json
# def geo_grab(st_address, city):
#     api_stem = 'http://where.yahooapis.com/geocode?'
#     params = {}
#     params['flags'] = 'J'
#     params['appid'] = 'ppp68N8k'
#     params['location'] = '%s %s'%(st_address,city)
#     url_params = urllib.parse.urlencode(params)
#     yahoo_api = api_stem + url_params
#     print('yahoo api:',yahoo_api)
#     c = urllib.request.urlopen(yahoo_api)
#     return json.loads(c.read())
# from time import sleep
# def mass_place_find(filename):
#     f = open('places2.txt','w')
#     for line in f.readlines():
#         line = line.strip()
#         line_arr = line.split('\t')
#         ret_dict = geo_grab(line[1],line[2])
#         if ret_dict['ResultSet']['Error'] == 0:
#             lat = float(ret_dict['ResultSet']['Results'][0]['latitude'])
#             lng = float(ret_dict['ResultSet']['Results'][0]['longitude'])
#             print('%s\t%f\t%f'%(line_arr[0],lat,lng))
#             f.write('%s\t%f\t%f'%(line,lat,lng))
#         else:
#             print('error fetching')
#         sleep(1)
#     f.close()
# geo_results = geo_grab('1 VA Center','Augusta, ME')
# print('geo_results:',geo_results)

KMEANS.clusterClubs(3)


print('yahoo api test end!')