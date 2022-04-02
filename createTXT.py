
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
freObj = ut.prequency_feature(1, 1000)
print(len(freObj))
freObj = str(freObj)

#txt 불러오기
testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
list = (readFile[1:-1].replace("'",'')).split(',')



#txt 로 저장
with open('freObj.txt', 'w') as file:
   file.writelines(freObj)
