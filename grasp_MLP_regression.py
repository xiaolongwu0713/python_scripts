from numpy import vstack
from numpy import sqrt
from sklearn.metrics import mean_squared_error
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 80)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        self.hidden2 = Linear(80, 40)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        self.hidden3 = Linear(40, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        self.hidden4 = Linear(10, 8)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
        self.lastlayer = Linear(8, 1)
        xavier_uniform_(self.lastlayer.weight)


    # forward propagate input
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.lastlayer(X)
        return X

learning_rate=0.01 # 0.02:failed to converge
# train the model
def train_model(train_dl, model):
    model.train()
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = SGD(model.parameters(), lr=learning_rate) # failed to converge
    for epoch in range(2000):
        # enumerate mini batches
        ls=.0
        for i, (inputs, targets) in enumerate(train_dl):
            yhat = model(inputs)
            loss = criterion(torch.squeeze(yhat), targets)
            ls=ls+loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i==19:
                print(f'epoch: {epoch:3} loss: {ls:10.8f}')
                plt.cla()
                ax.plot(targets.data.numpy(), color="orange")
                ax.plot(yhat.data.numpy(), 'g-', lw=3)
                plt.show()
                plt.pause(0.2)  # Note this correction
            if epoch % 100 == 0:
                pic= "train_%i.png" % epoch
                #plt.savefig(pic, format='png')

def evaluate_model(test_dl, model):
    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    plt.cla()
    ax.plot(actuals, color="orange")
    ax.plot(predictions, 'g-', lw=3)
    ax.set_title('Testing set', fontsize=35)
    plt.show()
    plt.pause(5)
    plt.savefig('test_sample.png')
    return mse

###########
datafile1='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
datafile2='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'
raw1=scipy.io.loadmat(datafile1)
raw2=scipy.io.loadmat(datafile2)
train=raw1['train'] # (6299, 115)
test=raw2['test'] #(2699, 115)
tmpraw=np.concatenate((train,test),0) # (8998, 115)

scaler = MinMaxScaler(feature_range=(-1, 1))
tmp = scaler.fit_transform(tmpraw)

#t = np.linspace(0, 20, 10000, endpoint=False)
#sig1=signal.square(2 * np.pi * 5 * t)[0:8998] + np.random.normal(1,2,8998)
#sig2=np.random.normal(1,2,8998)
#tmp=np.ones((8998,114))
#tmp[:,0]=sig1
#tmp[:,2]=sig2
#y=sig1 * 20- sig2 * 4.5;
#tmp=np.concatenate((tmp,np.expand_dims(y, axis=1)),1)

# train/test split
x=torch.FloatTensor(tmp[0:8400,0:-1]) #torch.Size([8998, 114])
y=torch.FloatTensor(tmp[0:8400,-1]) #torch.Size([8998])
x1=torch.FloatTensor(tmp[8400:,0:-1]) #torch.Size([8998, 114])
y1=torch.FloatTensor(tmp[8400:,-1]) #torch.Size([8998])

class dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)
#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))
train_data = dataset(x,y)
test_data = dataset(x1,y1)
train_loader = DataLoader(dataset=train_data, batch_size=299, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=299, shuffle=False)

# define the network
input_size=114
model = MLP(input_size)
fig, ax = plt.subplots(figsize=(12,7))
plt.ion()
# train the model
train_model(train_loader, model)
# evaluate the model
mse = evaluate_model(test_loader, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))