import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adagrad
import numpy as np
from torch.utils.data import Dataset, DataLoader
from grasp.feature3DToRealForce.model import modell
from grasp.feature3DToRealForce.utils import evaluate, train, dataset
from grasp.utils import read_fbanddata


basedir='/Users/long/BCI/python_scripts/grasp/feature3DToRealForce/'
resultdir=basedir+'result/'

#99 train trails, 5 test trials
trainx, trainy, testx, testy =read_fbanddata() # trainx: (114, 33293)

# input data shape: (trials,channels,timepoints)
trainx=trainx.swapaxes(1,2)
trainx=trainx.swapaxes(0,1) #(98, 180, 299)
trainy=trainy.swapaxes(0,1) #(98, 299)
testx=testx.swapaxes(1,2)
testx=testx.swapaxes(0,1) #(6, 180, 299)
testy=testy.swapaxes(0,1) #(6, 299)

#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))

train_data = dataset(trainx,trainy)
test_data = dataset(testx,testy)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

#wind=1000
#stride=500
wind=40 # 2s window
stride=1 # every 50ms
channlNum=trainx.shape[1]
kernLength1=wind//2
kernLength2=wind//2

#model = torch.load('model1.pth')
epochs = 1000
learning_rate = 0.0001
#model=torch.load(resultdir+'model_250.pth')
model = modell(channlNum,kernLength1,kernLength2)
#criterion = MSELoss().float()
criterion = MSELoss()
# Adam can't even descend the training error, but hold still !!! Don't understand this.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-4)
#optimizer = Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
for epoch in range(0,epochs):
    #train(model,train_loader,'plot')
    train(epoch,model,train_loader,optimizer,criterion,wind,stride,resultdir, plot='plot')
    if epoch % 1 == 0:
        #_ ,_ = evaluate(model,test_loader,'plot')
        _,_ = evaluate(epoch,model,test_loader,criterion,wind,stride,resultdir, plot='plot')
        modelfile= resultdir+ "model_%d.pth" % epoch
        torch.save(model, modelfile)
