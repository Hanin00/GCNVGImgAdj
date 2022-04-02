# idxid label에 따른 adj
import numpy as np
import sys
import json


np.set_printoptions(threshold=sys.maxsize)


testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
list = (readFile[1:-1].replace("'",'')).replace(' ','').split(',')

freObj = list[:100]

featureMatrix = []
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(1000):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
        objects = data[i]["objects"]
        for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
            object.append(objects[j]['names'])  # 이미지 하나에 대한 objList

       # object = sum(object, [])

        # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출

        row = [0 for i in range(len(freObj))]
        l = 0
        for k in range(len(freObj)) :
            for j in range(len(objects)):
                n = ''.join(object[j])
                m = freObj[k]

                if n in freObj :
                    w = freObj.index(n)
                    row[w] = 1

        featureMatrix.append((row))

featureMatrix =  np.array(featureMatrix)