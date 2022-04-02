
import sys
import csv
import util as ut
import YEmbedding as yed
import numpy as np
import pandas as pd














#
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
#
# print('freObj : ', freObj)
# print('adjMatrix : ', adjMatrix)
# # print('featuremap : ', featuremap)





#listitem = label[1]
#output_array = np.array(listitem)
#



# label txt 저장
# xlxspath = './data/image_regions.xlsx'
# # Y - image, cluser 몇 번인지~
# embedding_clustering = yed.YEmbedding(xlxspath)
# idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
# label = idCluster['cluster']
# j = label.tolist()
# print(type(j))
# print(j)
#
# list_a = list(map(str, j))
#
# #txt 로 저장
# with open('cluster.txt', 'w') as file:
#    file.writelines(','.join(list_a))
#


# txt 불러오기
# testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# list = (readFile[1:-1].replace("'",'')).split(',')


# id x id 동일 클러스터 Adj 저장
# #adjM = np.zeros((len(imgCnt), len(imgCnt))) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# adjM = np.zeros((1000, 1000)) #1000x1000(id 개수로 해야함. 근데 테스트라 10개만)
# for i in range(len(adjM[0])):
#     for j in range(len(adjM[0])):
#         if cLabel[i] == cLabel[j]:
#             adjM[i][j] += 1
#         if i == j:
#             adjM[i][j] += 1
#
#

# np.save('idAdj.npy',adjM)
#n = np.load('idAdj.npy')
print(n)