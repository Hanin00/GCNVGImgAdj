#ref https://docs.dgl.ai/en/0.6.x/tutorials/blitz/1_introduction.html

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn import GraphConv
#model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

#train
def train(features, labels, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    # features = g.ndata['feat']
    # labels = g.ndata['label']
    '''마스크 만들어야함. 랜덤으로 01 만들면 되나?'''


    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))




featuremap = np.load('./data/idFreFeature.npy')
idAdj = np.load('./data/idAdj.npy')
testFile = open('./data/cluster.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:-1].replace("'",'').replace(' ', '')).split(',')

labels = []
for i in range(1000)  :
    labels.append(int(label[i]))
labels = torch.tensor(labels)


print(featuremap.shape)
print(type(featuremap.shape))
print(type(featuremap.shape[1]))
print(featuremap.shape)
print(featuremap.shape[1])
print(labels)


#
# # Create the model with given dimensions
# #model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
# model = GCN(idAdj.shape, 16, label)
# train(featuremap, label, model)