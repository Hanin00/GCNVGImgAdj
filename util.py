import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
from gensim.models import FastText
from tqdm import tqdm
from collections import Counter
import YEmbedding
from visual_genome import api
import visual_genome_python_driver.visual_genome.local as lc




np.set_printoptions(linewidth=np.inf)

''' 1000개의 이미지의 빈출 objName '''
def prequency_feature(startId, endId):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(endId - startId):
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])
        object = sum(object, [])
        count_items = Counter(object)
        frqHundred = count_items.most_common(100)
        adjColumn = []
        for i in range(len(frqHundred)):
            adjColumn.append(frqHundred[i][0])

        return adjColumn



'''adjMatrix 생성
list = imgId에 대한 cluster 리스트
클러스터링 값이 같으면 += 1로 인접을 표현하고 자기 자신에 대해서도 1값을 추가함
'''
def create_adjMatrix(clusterList):
    adjM = np.zeros((len(clusterList), len(clusterList)))
    for i in range(len(clusterList)) :
        for j in range(len(clusterList)) :
            if clusterList[i] == clusterList[j] :
                adjM[i][j] += 1
            if i==j :
                adjM[i][j] += 1
    return adjM

''' imageFeature 생성(이미지 하나에 대한) '''
def featuremap(startId, endId, freObjList ) :
    featureMatrix=[]
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(endId-startId):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])  # 이미지 하나에 대한 objList

            # object = sum(object, [])

            # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출

            row = [0 for i in range(len(freObjList))]
            l = 0
            for k in range(len(freObjList)):
                for j in range(len(objects)):
                    n = ''.join(object[j])
                    m = freObjList[k]

                    if n in freObjList:
                        w = freObjList.index(n)
                        row[w] = 1

            featureMatrix.append((row))
    featureMatrix=np.array(featureMatrix)

    return featureMatrix



''' adj 생성(이미지 하나에 대한) '''
def createAdj(imageId, adjColumn):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])

        adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))
        data_df = pd.DataFrame(adjMatrix)
        data_df.columns = adjColumn
        data_df = data_df.transpose()
        data_df.columns = adjColumn

        # ralationship에 따른
        for q in range(len(object)):
            row = adjColumn.index(object[q])
            column = adjColumn.index(subject[q])
            adjMatrix[column][row] += 1

        return data_df, adjMatrix
