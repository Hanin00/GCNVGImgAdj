import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import utils

from dgl.nn import GraphConv




#
# criterion = nn.CrossEntropyLoss()
# class GCN_layer(nn.Module):
#     def __init__(self, in_features, out_features, A):
#         super(GCN_layer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.A = A
#         self.fc = nn.Linear(in_features, out_features)
#
#     # A와 X의 곱으로 얻어진 매트릭스를 row단위로 FCLayer에 넘김
#     # A : 모든 논문의 그래프 정보가 담긴 2708x2708
#     # X : 각 논문의 feature 정보가 담긴 2708x1433 매트릭스
#     def forward(self, X):
#         return self.fc(torch.spmm(self.A, X))  # 이웃 정보 종합
#
#
# # GCN_layer를 두 번 거침 -> A*(embedded(AX)) 행렬을 얻음
# # -> GCN결과로 나온 행렬의 (1,1)값 = 논문 1과 이웃한 논문들의 feature + 이웃한 논문의 이웃 논문 feature 정보
# class GCN(nn.Module):
#     def __init__(self, num_feature, num_class, A):
#         super(GCN, self).__init__()
#
#         self.feature_extractor = nn.Sequential(
#             GCN_layer(num_feature, 16, A),
#             nn.ReLU(),
#             GCN_layer(16, num_class, A)
#         )
#
#     def forward(self, X):
#         return self.feature_extractor(X)
#
