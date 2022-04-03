import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#from mTest.utils import load_data, accuracy
#from mTest.models import GCN


import utils

#dataLoad
# path = 'C:/Semester2201/LAB/GNN/CoroEx/pygcn/data/cora/'
# dataset = 'cora'
#
# A, features, labels, idx_train, idx_val, idx_test = utils.load_data(path, dataset)

#from main import idx_train, labels, features, idx_val, idx_test


featuremap = np.load('./data/idFreFeature.npy')
idAdj = np.load('./data/idAdj.npy')
testFile = open('./data/freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:-1].replace("'",'')).replace(' ','').split(',')




criterion = nn.CrossEntropyLoss()

def train(model, Loss, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []

    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 10.

    for epoch in range(num_epochs):

        # Forward Pass
        model.train()
        output = model(features)
        train_loss = criterion(output[idx_train], labels[idx_train])

        # Backward and optimize
        train_loss.backward()
        optimizer.step()

        train_loss_arr.append(train_loss.data)

        if epoch % 10 == 0:
            model.eval()

            output = model(features)
            val_loss = criterion(output[idx_val], labels[idx_val])
            test_loss = criterion(output[idx_test], labels[idx_test])

            val_acc = utils.accuracy(output[idx_val], labels[idx_val])
            test_acc = utils.accuracy(output[idx_test], labels[idx_test])

            test_loss_arr.append(test_loss)

            if best_ACC < val_acc:
                best_ACC = val_acc
                early_stop = 0
                final_ACC = test_acc
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

    print("Final Accuracy::", final_ACC)
