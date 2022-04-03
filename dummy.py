#https://chioni.github.io/posts/gnn/#node-classification-task-example

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import util2 as ut2

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


#TRAIN
def train(model, Loss, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []
    best_ACC = 0
    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 10.
    for epoch in range(num_epochs):
        # Forward Pass
        model.train()
        output = model(features)
        #model, target
      #  train_loss = criterion(output[idx_train],labels[idx_train])
        train_loss = Loss(output[idx_train],labels[idx_train])

        # Backward and optimize
        train_loss.backward()
        optimizer.step()

        train_loss_arr.append(train_loss.data)

        if epoch % 10 == 0:
            model.eval()
            output = model(features)
            val_loss = criterion(output[idx_val], labels[idx_val])
            test_loss = criterion(output[idx_test], labels[idx_test])

            # val_acc = ut2.accuracy(output[idx_val], labels[idx_val])
            # test_acc = ut2.accuracy(output[idx_test], labels[idx_test])
            val_acc = ut2.accuracy(output[idx_val], labels[idx_val])
            test_acc = ut2.accuracy(output[idx_test], labels[idx_test])

            test_loss_arr.append(test_loss)

            if best_ACC < val_acc:
                best_ACC = val_acc
                early_stop = 0

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test ACC: {:.4f} *'.format(epoch, 100,
                                                                                                        train_loss.data,
                                                                                                        test_loss,
                                                                                                      test_acc))
            else:
                early_stop += 1

                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test ACC: {:.4f}'.format(epoch, 100,
                                                                                                      train_loss.data,
                                                                                                      test_loss,
                                                                                                      test_acc))

        if early_stop >= early_stop_max:
            break
    final_ACC = test_acc
    print("Final Accuracy::", final_ACC)


# TEST
def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))




features = np.load('./data/idFreFeature.npy')
A = torch.Tensor(np.load('./data/idAdj.npy'))
testFile = open('./data/cluster.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = readFile[1:].replace("'",'').replace(' ','').split(',')
features = torch.Tensor(features)
labels = []
for i in range(1000)  :
    labels.append(int(label[i]))
labels = torch.Tensor(labels)
labels = labels.long()
idx_train = range(0,400)
idx_val =(400,700)
idx_test = (700,999)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


# GCN 학습 돌려서 epoch에 따른 Loss 확인
model = GCN(features.size(1), labels.size(0), A)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)

#train(model, criterion, optimizer, 1000)
train(model, criterion, optimizer, 1000)


