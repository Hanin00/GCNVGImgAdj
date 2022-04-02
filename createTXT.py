import sys
import csv
import util as ut
import YEmbedding as yed
import numpy as np
import pandas as pd
import json



# startId = 1
# endId = 10
# xlxspath = './data/image_regions.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']

# print(label)
# freObj = ut.prequency_feature(1, 1000)
# adjMatrix = ut.create_adjMatrix(clusterList=label[1])
# # featuremap =ut.featuremap(startId,endId,freObj)



#listitem = label[1]
#output_array = np.array(listitem)

'''label txt 저장'''
# xlxspath = './data/image_regions.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# j = label.tolist()
# print(type(j))
# print(j)
# list_a = list(map(str, j))
#
'''txt 로 저장'''
# with open('cluster.txt', 'w') as file:
#    file.writelines(','.join(list_a))
#
# 
''' 1000개의 이미지의 최빈 objName 100개'''
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#     object = []
#     for i in range(1000):
#         objects = data[i]["objects"]
#         for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
#             object.append(objects[j]['names'])
#     object = sum(object, [])
#     count_items = Counter(object)
#     frqHundred = count_items.most_common(100)
#     adjColumn = []
#     for i in range(len(frqHundred)):
#         adjColumn.append(frqHundred[i][0])
# 
#     with open('freObj.txt', 'w') as file:
#        file.writelines(','.join(adjColumn))
# 
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).split(',')
# 
# print(len(list))
# print(list[0])




'''txt 불러오기'''
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).split(',')

'''id x id 동일 클러스터 Adj 저장'''
# #adjM = np.zeros((len(imgCnt), len(imgCnt))) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# adjM = np.zeros((1000, 1000)) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# for i in range(len(adjM[0])):
#     for j in range(len(adjM[0])):
#         if cLabel[i] == cLabel[j]:
#             adjM[i][j] += 1
#         if i == j:
#             adjM[i][j] += 1
# np.save('idAdj.npy',adjM)
#idAdj = np.load('idAdj.npy')

#
# '''img 당 freObj 있/없 featuremap'''
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).replace(' ','').split(',')
#
# freObj = list[:100]
#
# featureMatrix = []
# with open('./data/scene_graphs.json') as file:  # open json file
#     data = json.load(file)
#     object = []
#     for i in range(1000):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
#         objects = data[i]["objects"]
#         for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
#             object.append(objects[j]['names'])  # 이미지 하나에 대한 objList
#
#        # object = sum(object, [])
#
#         # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
#
#         row = [0 for i in range(len(freObj))]
#         l = 0
#         for k in range(len(freObj)) :
#             for j in range(len(objects)):
#                 n = ''.join(object[j])
#                 m = freObj[k]
#
#                 if n in freObj :
#                     w = freObj.index(n)
#                     row[w] = 1
#
#         featureMatrix.append((row))
# featureMatrix =  np.array(featureMatrix)
# np.save('idFreFeature.npy',featureMatrix)
# featuremap = np.load('idFreFeature.npy')
