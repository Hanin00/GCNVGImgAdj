import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


#from mTest.utils import load_data, accuracy
#from mTest.models import GCN


#dataLoad
# path = 'C:/Semester2201/LAB/GNN/CoroEx/pygcn/data/cora/'
# dataset = 'cora'
#
# A, features, labels, idx_train, idx_val, idx_test = utils.load_data(path, dataset)

#from main import idx_train, labels, features, idx_val, idx_test


'''featuremap = np.load('./data/idFreFeature.npy')
idAdj = np.load('./data/idAdj.npy')
testFile = open('./data/freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:-1].replace("'",'')).replace(' ','').split(',')

'''

