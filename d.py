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



with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(9):
        objects = data[i]["objects"]
        for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
            object.append(objects[j]['names'])
    object = sum(object, [])
    count_items = Counter(object)
    frqHundred = count_items.most_common(10)
    print(len(frqHundred))


    adjColumn = []
    for i in range(len(frqHundred)):
        adjColumn.append(frqHundred[i][0])

    print(adjColumn)