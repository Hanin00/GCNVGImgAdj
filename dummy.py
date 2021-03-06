#https://chioni.github.io/posts/gnn/#node-classification-task-example

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import util2 as ut2
import train, model


#TRAIN
def train(model, Loss, optimizer, num_epochs):
    train_loss_arr = []
    test_loss_arr = []
    best_ACC = 0.0
    best_test_loss = 99999999
    early_stop, early_stop_max = 0., 10.
    for epoch in range(num_epochs):
        # Forward Pass
        model.train() #모델을 학습 모드로 변환
        output = model(features)  # data를 모델에 넣어서 가설(hypothesis) 획득
        #model, target
      #  train_loss = criterion(output[idx_train],labels[idx_train])
        train_loss = Loss(output[idx_train],labels[idx_train])

        # Backward and optimize
        train_loss.backward() # loss를 역전파 알고리즘으로 계산
        optimizer.step() # 계산한 기울기를 앞서 정의한 알고리즘에 맞춰 가중치를 수정

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


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))







features = np.load('./data/idFreFeature.npy')
A = torch.Tensor(np.load('./data/idAdj.npy'))
testFile = open('./data/cluster.txt','r')
readFile = testFile.readline()
label = readFile[1:].replace("'",'').replace(' ','').split(',')
features = torch.Tensor(features)
labels = []
for i in range(1000) :
    labels.append(int(label[i]))
labels = torch.Tensor(labels)
labels = labels.long()


#train/val/text
idx_train = range(0,400)
idx_val =(400,700)
idx_test = (700,999)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)





# GCN 학습 돌려서 epoch에 따른 Loss 확인
model = model.GCN(features.size(1), labels.size(0), A)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)

#train(model, criterion, optimizer, 1000)
train(model, criterion, optimizer, 1000)

