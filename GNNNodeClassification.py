#https://github.com/fahim-sikder/Node-Classification-GCN/blob/master/pytorch-GCN.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import util2 as ut2
import train, model
import numpy as np
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tnrange
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from pathlib import Path
from functools import partial


'''
GNN Node Classification
dataset : Visual Genome.Scence graph 
 
Adj : id x id (같은 cluster 값을 갖는 경우 1로 체크함), 1000x1000
Feature : id x freOBJ(해당 img에 FreOBJ가 있는 경우 1, 없는 경우 0)
freObj : 대상이 되는 이미지 1000개의 Scene graph에서 가장 언급량이 많은 Obj 100개
Label : 각 id 당 cluster 번호
cluster는 bert-base-nli-mean-tokens를 이용해 15개의 클러스터로 분류했음
-> 1000개의 region_graph의 phrase 값을 embedding 함
-> 이 부분도 relationship이 잘 나타나지 않은 것 같아 아쉬움
relationship을 더 잘 활용할 수 있는걸 하고싶음..

'''





#gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



features = csr_matrix(np.load('./data/idFreFeature.npy'),dtype=np.float32)
adj  = torch.FloatTensor(np.load('./data/idAdj.npy'))
features = csr_matrix(features)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

features = normalize(features)
testFile = open('./data/cluster.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'",'').replace(' ','').split(','))
labels = []
for i in range(1000)  :
    labels.append(int(label[i]))

features = torch.FloatTensor(features.todense())   #<class 'torch.Tensor'>
labels = torch.LongTensor(labels)  #<class 'torch.Tensor'>


# dataset train/test/val로 나눔
np.random.seed(34)
n_train = 200
n_val = 300
n_test = len(features) - n_train - n_val
idxs = np.random.permutation(len(features))
idx_train = torch.LongTensor(idxs[:n_train])
idx_val   = torch.LongTensor(idxs[n_train:n_train+n_val])
idx_test  = torch.LongTensor(idxs[n_train+n_val:])

#cuda.. gpu로 보냄
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)



# Model
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight를 reset해 줌. weight 변화에 따른 정확도 측정을 위해서? 왜하지? 이유 찾아보기
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        #

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # X : 초기 랜덤값 -> 학습 하면서 변경
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]      # 100

#seed 고정
torch.manual_seed(34)

# model
model = GCN(nfeat=n_features,
            nhid=20,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

def step():
    t = time.time()
    model.train()  #model 학습모드로
    optimizer.zero_grad()
    output = model(features, adj)  #model에 값 넣음
    loss = F.nll_loss(output[idx_train], labels[idx_train]) #loss 함수
    acc = accuracy(output[idx_train], labels[idx_train]) #accuracy 파악
    loss.backward()
    optimizer.step()

    return loss.item(), acc

#평가
def evaluate(idx):
    model.eval()
    output = model(features, adj) #모델 돌림
    loss = F.nll_loss(output[idx], labels[idx]) #모델이 분류한 값과 label 비교해서 loss 파악
    acc = accuracy(output[idx], labels[idx])

    return loss.item(), acc

#정확도
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 비슷하다고 뽑은 것들중에 제일 비슷한 거
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []

for i in tnrange(epochs):
    tl, ta = step()
    train_loss += [tl]
    train_acc += [ta]

    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]

        print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))


output = model(features, adj)


samples = 10
#torch.randperm : 데이터 셔플
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]

idx2lbl = ['1번 그림', '2번 그림', '3번 그림', '4번 그림', '5번 그림', '6번 그림', '7번 그림'
, '8번 그림', '9번 그림', '10번 그림', '11번 그림', '12번 그림', '13번 그림', '14번 그림','15번 그림']
#실제 라벨 10개, predicate 결과 10개
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
print(df)







# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# path = Path('./data/cora')
#
# paper_features_label = np.genfromtxt(path/'cora.content', dtype=np.str)
#
# features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
# labels = paper_features_label[:, -1]
# lbl2idx = {k:v for v,k in enumerate(sorted(np.unique(labels)))}
# labels = [lbl2idx[e] for e in labels]
#
# papers = paper_features_label[:,0].astype(np.int32)
#
# paper2idx = {k:v for v,k in enumerate(papers)}
# edges = np.genfromtxt(path/'cora.cites', dtype=np.int32)
# edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
#
# adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                  shape=(len(labels), len(labels)), dtype=np.float32)
#
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
# def normalize(mx):
#     rowsum = np.array(mx.sum(1))
#     r_inv = (rowsum ** -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#
# features = normalize(features)
#
# #eye : 단위행렬 만드는 함수
# adj = normalize(adj + eye(adj.shape[0])) #Normalizing for removing gradient vanishing and exploding problem
# #normarlize(원래 adj + 단위행렬(자기 자신 가리킴))
#
# #todensd() : Scipy 압축 희소행(CSR) 행렬을 Numpy() 행렬로 변환
# adj = torch.FloatTensor(adj.todense())
#
# features = torch.FloatTensor(features.todense()) #scipy.sparse -> numpy.matrix->torch.tensor
# labels = torch.LongTensor(labels)
#
#
#
# # dataset train/test/val로 나눔
# np.random.seed(34)
# n_train = 200
# n_val = 300
# n_test = len(features) - n_train - n_val
# idxs = np.random.permutation(len(features))
# idx_train = torch.LongTensor(idxs[:n_train])
# idx_val   = torch.LongTensor(idxs[n_train:n_train+n_val])
# idx_test  = torch.LongTensor(idxs[n_train+n_val:])
#
# #cuda.. gpu로 보냄
# adj = adj.to(device)
# features = features.to(device)
# labels = labels.to(device)
# idx_train = idx_train.to(device)
# idx_val = idx_val.to(device)
# idx_test = idx_test.to(device)
#
# # Model
# class GraphConvolution(Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         # weight를 reset해 줌. weight 변화에 따른 정확도 측정을 위해서? 왜하지? 이유 찾아보기
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#         #
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ') '
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     # X : 초기 랜덤값 -> 학습 하면서 변경
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
#
#
# n_labels = labels.max().item() + 1  # 7
# n_features = features.shape[1]      # 1433
#
# #seed 고정
# torch.manual_seed(34)
#
# # model
# model = GCN(nfeat=n_features,
#             nhid=20,  # hidden = 16
#             nclass=n_labels,
#             dropout=0.5)  # dropout = 0.5
# optimizer = optim.Adam(model.parameters(),
#                        lr=0.001, weight_decay=5e-4)
#
# def step():
#     t = time.time()
#     model.train()  #model 학습모드로
#     optimizer.zero_grad()
#     output = model(features, adj)  #model에 값 넣음
#     loss = F.nll_loss(output[idx_train], labels[idx_train]) #loss 함수
#     acc = accuracy(output[idx_train], labels[idx_train]) #accuracy 파악
#     loss.backward()
#     optimizer.step()
#
#     return loss.item(), acc
#
# #평가
# def evaluate(idx):
#     model.eval()
#     output = model(features, adj) #모델 돌림
#     loss = F.nll_loss(output[idx], labels[idx]) #모델이 분류한 값과 label 비교해서 loss 파악
#     acc = accuracy(output[idx], labels[idx])
#
#     return loss.item(), acc
#
# #정확도
# def accuracy(output, labels):
#     preds = output.max(1)[1].type_as(labels) # 비슷하다고 뽑은 것들중에 제일 비슷한 거
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)
#
#
# epochs = 1000
# print_steps = 100
# train_loss, train_acc = [], []
# val_loss, val_acc = [], []
#
# for i in tnrange(epochs):
#     tl, ta = step()
#     train_loss += [tl]
#     train_acc += [ta]
#
#     if ((i + 1) % print_steps) == 0 or i == 0:
#         tl, ta = evaluate(idx_train)
#         vl, va = evaluate(idx_val)
#         val_loss += [vl]
#         val_acc += [va]
#
#         print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))
#
# '''
# fig, axes = plt.subplots(1,2, figsize=(15,5))
#
# ax = axes[0]
# axes[0].plot(train_loss[::print_steps] + [train_loss[-1]], label='Train')
# axes[0].plot(val_loss, label='Validation')
# axes[1].plot(train_acc[::print_steps] + [train_acc[-1]], label='Train')
# axes[1].plot(val_acc, label='Validation')
# axes[0].grid()
# axes[1].grid()
#
# for ax, t in zip(axes, ['Loss','Accuracy']):
#     ax.legend(), ax.set_title(t, size=15)'''
#
# output = model(features, adj)
# print("output : ",output)
#
# samples = 10
# #torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
#
# idx2lbl = {v:k for k,v in lbl2idx.items()}
# #실제 라벨 10개, predicate 결과 10개
# df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
#                    'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
# print(df)
