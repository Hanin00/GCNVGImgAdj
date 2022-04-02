import json
from collections import Counter

'''featureMatrix = []
with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(3 - 1):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
        objects = data[i]["objects"]
        for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
            object.append(objects[j]['names'])
        object = sum(object, [])  # 이미지 하나에 대한 objList
        print(object)
        print(type(object))
'''

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(10):   #최빈 단어를 뽑을 대상이 되는 이미지의 개수
        objects = data[i]["objects"]
        for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
            object.append(objects[j]['names'])
    object = sum(object, [])
    count_items = Counter(object)
    print("count_items",len(count_items))
    frqHundred = count_items.most_common(100)
    print("frqHundred",len(frqHundred))
    adjColumn = []
    for i in range(len(frqHundred)):
        adjColumn.append(frqHundred[i][0])
    print(len(adjColumn))
    print("adjColumn", adjColumn)