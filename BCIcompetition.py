import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

n_channels = 64
sampling_rate = 1000
batch_size=4
learning_rate = 0.001
epochs=35

trainfile='data/BCI-competition/Competition_train.mat'
train = scipy.io.loadmat(trainfile)
trainx=train['X'] #(278, 64, 3000)
trainy=train['Y'] #(278, 1)
trainy=(trainy+1)/2

testfile='data/BCI-competition/Competition_test.mat'
test = scipy.io.loadmat(testfile)
testx=test['X'] #(100, 64, 3000)
true_labels_file='data/BCI-competition/true_labels.txt'
testy=np.loadtxt(true_labels_file)
testy=((testy.astype(int)+1)/2).astype(int)
testy=torch.tensor(testy)
#os.getcwd()


## train data
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)
#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))
train_data = trainData(torch.from_numpy(trainx.astype(np.float32)),torch.from_numpy(trainy.astype(np.float32)))
train_loader = DataLoader(dataset=list(iter(train_data))[0:8], batch_size=batch_size, shuffle=True)
#train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# batch=iter(train_loader).next()
# batch[0].shape: torch.Size([4, 64, 3000])
# batch[1].shape: torch.Size([4, 1])
# train=list(iter(train_data)) # indexing the dataloader

## test data
class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__(self):
        return len(self.X_data)
#test_data = testData(torch.from_numpy(testx.astype(np.float32)))
# use some train data as test data
test_loader = DataLoader(dataset=list(iter(train_data))[200:208], batch_size=batch_size)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.lstm=nn.LSTM(64,100,2) # (feature_size,hidden_size, layers)
        self.linear1 = nn.Linear(100, 10) #
        self.linear2 = nn.Linear(10, 1)  #
        self.relu = nn.ReLU()
        # self.batchnorm = nn.BatchNorm1d(100) # better not using

    def forward(self, inputs):
        x, _ = self.lstm(inputs)
        #x = self.batchnorm(x[-1,:,:])
        x = self.linear1(x[-1,:,:])
        x = self.relu(x)
        x = self.linear2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = binaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch=X_batch.permute(2,0,1) # change from (4,64,3000) to (3000 time_steps,4 batch_size,64 features)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(torch.squeeze(y_pred), torch.squeeze(y_batch))
        acc = binary_acc(torch.squeeze(y_pred), torch.squeeze(y_batch))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    y_pred_list = []
    testy = []
    model.eval()
    with torch.no_grad():
        for X_batch,test_y in test_loader:
            X_batch = X_batch.permute(2, 0, 1)
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_list = np.append(y_pred_list, y_test_pred.cpu().numpy()) # should append to form a 1D numpy
            testy = np.append(testy, test_y.numpy())

    acc = binary_acc(torch.squeeze(torch.tensor(y_pred_list)), torch.squeeze(torch.tensor(testy)))
    print(f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Test Acc: {acc}:.3f')


acc = binary_acc(torch.squeeze(torch.tensor(y_pred_list)), torch.squeeze(testy))
#y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#predy=np.array(y_pred_list)
#predy=predy.astype(int)
#compare=np.array([i==j for i,j in zip(predy, testy)]).astype(int)
#acc=compare.sum()/len(compare)

#torch.save(model, 'model_BCIcompetition.pth')
#model = torch.load('model_BCIcompetition.pth')

confusion_matrix(testy, y_pred_list)
print(classification_report(testy, y_pred_list))

### evaluate while training ####
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.permute(2, 0, 1)  # change from (4,64,3000) to (3000 time_steps,4 batch_size,64 features)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(torch.squeeze(y_pred), torch.squeeze(y_batch))
        acc = binary_acc(torch.squeeze(y_pred), torch.squeeze(y_batch))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.permute(2, 0, 1)
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    predy = np.array(y_pred_list)
    predy = predy.astype(int)
    compare = np.array([i == j for i, j in zip(predy, testy)]).astype(int)
    acc = compare.sum() / len(compare)

    print(
        f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f} | Test accuracy:{acc:.3f}')

