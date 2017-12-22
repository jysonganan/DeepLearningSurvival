from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from torch.nn.parameter import Parameter

D_in, H, D_out = 1000, 128, 32
batch_size = 32
num_time_units = 24   # 24 months
time_bin = 30
n_epochs = 20
learning_rate = 1e-3

class survdl(nn.Module):
    def __init__(self, D_in, H, D_out, num_time_units):
        super(survdl, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc_layer = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Dropout(0.5), nn.Linear(H, D_out))
        self.fc_layer2 = nn.Linear(1, num_time_units)
        self.beta = Parameter(torch.Tensor(D_out, 1))
        self.beta.data.uniform_(-0.001, 0.001)

    
    def score_1(self, x):
        return torch.exp(x.mm(self.beta))
                
    def score_2(self, score1):
        return self.sigmoid(self.fc_layer2(score1))
   
    def forward(self, x):
        new_x = self.fc_layer(x)
        score1 = self.score_1(new_x)
        score2 = self.score_2(score1)
        return score1, score2
        
def unique_set(lifetime):
    a = lifetime.data.cpu().numpy()
    t, idx = np.unique(a, return_inverse=True)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return t, unq_idx
    

def log_parlik(lifetime, censor, score1):  
    t, H = unique_set(lifetime)
    keep_index = np.nonzero(censor.data.cpu().numpy())[0]  #censor = 1
    H = [list(set(h)&set(keep_index)) for h in H]
    n = [len(h) for h in H]
    
    score1 = score1.data.cpu().numpy()
    total = 0
    for j in range(len(t)):
        total_1 = np.sum(np.log(score1)[H[j]])
        m = n[j]
        total_2 = 0
        for i in range(m):
            subtotal = np.sum(score1[sum(H[j:],[])]) - (i*1.0/m)*(np.sum(score1[H[j]]))
            subtotal = np.log(subtotal)
            total_2 = total_2 + subtotal
        total = total + total_1 - total_2
        total = np.array([total])
    return Variable(torch.from_numpy(total).type(torch.FloatTensor)).cuda().view(-1,1)
        

def acc_pairs(censor, lifetime):
    noncensor_index = np.nonzero(censor.data.cpu().numpy())[0]
    lifetime = lifetime.data.cpu().numpy()
    acc_pair = []
    for i in noncensor_index:
        all_j =  np.array(range(len(lifetime)))[lifetime > lifetime[i]]
        acc_pair.append([(i,j) for j in all_j])
    
    acc_pair = reduce(lambda x,y: x + y, acc_pair)
    return acc_pair


def rank_loss(lifetime, censor, score2, t, time_bin): 
    # score2 (n(samples)*24) at time unit t = 1,2,...,24
    acc_pair = acc_pairs(censor, lifetime)
    lifetime = lifetime.data.cpu().numpy()
    total = 0
    for i,j in acc_pair:
        yi = (lifetime[i] >= (t-1) * time_bin) * 1
        yj = (lifetime[j] >= (t-1) * time_bin) * 1
        a = Variable(torch.ones(1)).type(torch.FloatTensor).cuda()
        L2dist = torch.dist(score2[j, t-1] - score2[i, t-1], a, 2)
        total = total + L2dist* yi * (1-yj)
    return total


def C_index(censor, lifetime, score1):
    score1 = score1.data.cpu().numpy()
    acc_pair = acc_pairs(censor, lifetime)
    prob = sum([score1[i] >= score1[j] for (i, j) in acc_pair])[0]*1.0/len(acc_pair)
    return prob
    

model = survdl(D_in, H, D_out, num_time_units)
model.cuda()  

optimizer = optim.Adam(model.parameters(), lr = learning_rate)




def train(epoch):
    model.train()
    train_loss = 0    
    idx = np.random.permutation(X_train.shape[0])     
    j = 0
    while j < X_train.shape[0]:
        if j < X_train.shape[0] - batch_size:
            data = Variable(torch.from_numpy(X_train[idx[j:(j + batch_size)]])).type(torch.FloatTensor).cuda()
            lifetime = Variable(torch.from_numpy(Y_train[idx[j:(j + batch_size)],1])).type(torch.FloatTensor).cuda()
            censor = Variable(torch.from_numpy(Y_train[idx[j:(j + batch_size)],0])).type(torch.FloatTensor).cuda()
        else:
            data = Variable(torch.from_numpy(X_train[idx[j:]])).type(torch.FloatTensor).cuda()
            lifetime = Variable(torch.from_numpy(Y_train[idx[j:],1])).type(torch.FloatTensor).cuda()
            censor = Variable(torch.from_numpy(Y_train[idx[j:],0])).type(torch.FloatTensor).cuda()
            
        optimizer.zero_grad()
        score1, score2 = model(data)
        loss1 = log_parlik(lifetime, censor, score1)
        loss2 = []
        for t in range(num_time_units):
            loss2.append(rank_loss(lifetime, censor, score2, t+1, time_bin))
        loss2 = sum(loss2)
        loss = 1.0 * loss1 + 0.5 * loss2
        loss.backward()      
        train_loss = loss.data[0]
        optimizer.step()
        j += batch_size
    return train_loss*1.0 / X_train.shape[0]



def test(epoch):
    model.eval()
    test_loss = 0
    j = 0
    while j < X_test.shape[0]:
        if j < X_test.shape[0] - batch_size:
            data = Variable(torch.from_numpy(X_test[j:(j + batch_size)])).type(torch.FloatTensor).cuda()
            lifetime = Variable(torch.from_numpy(Y_test[j:(j + batch_size),1])).type(torch.FloatTensor).cuda()
            censor = Variable(torch.from_numpy(Y_train[idx[j:(j + batch_size)],0])).type(torch.FloatTensor).cuda()
        else:
            data = Variable(torch.from_numpy(X_test[j:])).type(torch.FloatTensor).cuda()
            lifetime = Variable(torch.from_numpy(Y_test[j:,1])).type(torch.FloatTensor).cuda()
            censor = Variable(torch.from_numpy(Y_train[idx[j:],0])).type(torch.FloatTensor).cuda()
        y_pred = model(data)
        score1, score2 = model(data)
        loss1 = log_parlik(lifetime, censor, score1)
        loss2 = []
        for t in range(num_time_units):
            loss2.append(rank_loss(lifetime, censor, score2, t+1, time_bin))
        loss2 = sum(loss2)
        loss = 1.0 * loss1 + 0.5 * loss2
        test_loss += loss.data[0]
        j += batch_size
    return test_loss*1.0 / X_test.shape[0]
        
    
    
for epoch in range(1, n_epochs + 1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    print('====> Epoch: %d training loss: %.4f'%(epoch, train_loss))
    print('====> Epoch: %d testing loss: %.4f'%(epoch, test_loss))
    
    
# concordance - training
data_train = Variable(torch.from_numpy(X_train)).type(torch.FloatTensor).cuda()
lifetime_train = Variable(torch.from_numpy(Y_train[:,0])).type(torch.FloatTensor).cuda()
censor_train = Variable(torch.from_numpy(Y_train[:,1])).type(torch.FloatTensor).cuda()

score1_train, score2_train = model(data_train)
C_index_train = C_index(censor_train, lifetime_train, score1_train)
print('Concordance index for training data: {:.4f}'.format(C_index_train))


# concordance - test
data_test = Variable(torch.from_numpy(X_test)).type(torch.FloatTensor).cuda()
lifetime_test = Variable(torch.from_numpy(Y_test[:,0])).type(torch.FloatTensor).cuda()
censor_test = Variable(torch.from_numpy(Y_test[:,1])).type(torch.FloatTensor).cuda()

score1_test, score2_test = model(data_test)
C_index_test = C_index(censor_test, lifetime_test, score1_test)
print('Concordance index for test data: {:.4f}'.format(C_index_test))
