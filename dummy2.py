# https://docs.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-train-model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import util2 as ut2

# Function to save the model
def saveModel():
    path = "./GCNModel.pth"
    torch.save(model.state_dict(), path)



#MODEL
class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, X): # A와 X의 곱으로 얻어진 매트릭스를 row 단위로 fully connected layer로 넘김
        #A : 모든 논문의 그래프 정보가 담긴 매트릭스  1000x1000
        #X : 각 논문의 feature 정보가 담긴 매트릭스  1000x100
        return self.fc(torch.spmm(self.A, X))  # 이웃 정보 종합


class GCN(nn.Module):
    def __init__(self, num_feature, num_class, A):
        super(GCN, self).__init__()

        self.feature_extractor = nn.Sequential(
            GCN_layer(num_feature, 16, A), #16 : 임의로 설정한 임베딩의 크기. 1000x16이 될 것
            nn.ReLU(),
            GCN_layer(16, num_class, A)
        )

    def forward(self, X):
        return self.feature_extractor(X)
