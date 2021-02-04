from config import *
import os
import numpy as np
import scipy.io
import re
import matplotlib.pyplot as plt

## usage: aa=loadData(31,1,'good'/'bad'/'power'/'epoch61'), then load the data part by checking aa.keys()
def loadData(pn, session, *args):
    arg=args[0]
    if len(args)==0:
        filename = os.path.join(raw_data,'P'+str(pn), '1_Raw_Data_Transfer',
                                'P'+str(pn)+'_H'+str(mode)+'_'+str(session)+'_Raw.mat')
    else:
        if arg=='good':
            filename=os.path.join(processed_data,'P'+str(pn),
                                  'P'+str(pn)+'_H'+str(mode)+'_'+str(session)+'_goodLineChn.mat')
        elif arg=='bad':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                    'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_goodLineChn.mat')
        elif arg == 'discard':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_discardChn.mat')
        elif arg == 'power':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_power.mat')
        elif arg == 'events':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable.txt')
            tmp=a=np.loadtxt(filename)
            return tmp
        elif arg == 'eventave':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable_ave.mat')
        elif arg == 'eventave':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable_ave.mat')
        elif re.compile('epoch').match(arg):
            filename=os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_' + arg+'.mat')
    mat = scipy.io.loadmat(filename)
    return mat  # return np arrary. avedata is the key of this dict, data dim: eles,time,trials

def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)


def read_fbanddata():
    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TrainData.mat'
    datafile11 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TestData.mat'
    datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TrainData.mat'
    datafile21 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TestData.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TrainData.mat'
    datafile31 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TestData.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
    datafile41 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'
    data1 = scipy.io.loadmat(datafile1)
    data11 = scipy.io.loadmat(datafile11)
    data2 = scipy.io.loadmat(datafile2)
    data21 = scipy.io.loadmat(datafile21)
    data3 = scipy.io.loadmat(datafile3)
    data31 = scipy.io.loadmat(datafile31)
    data4 = scipy.io.loadmat(datafile4)
    data41 = scipy.io.loadmat(datafile41)

    traintmp1 = data1['train']
    traintmp11= data11['test']
    traintmp2 = data2['train']
    traintmp21 = data21['test']
    traintmp3 = data3['train']
    traintmp31 = data31['test']
    traintmp4 = data4['train']
    traintmp41 = data41['test']

    dataset=np.concatenate((traintmp1,traintmp11,traintmp2,traintmp21,traintmp3,traintmp31,traintmp4,traintmp41),axis=0) # (40192, 115)

    testx=np.transpose(dataset[-1500:,:-1]) #(114, 1500)
    testy=np.transpose(dataset[-1500:,-1]) # (1500,)
    trainx=np.transpose(dataset[0:-1500,:-1]) #  (114, 38692)
    trainy=np.transpose(dataset[:-1500,-1]) # (38692,)


    del data1,data11,data2,data21,data3,data31,data4,data41
    del traintmp1, traintmp11,traintmp2,traintmp21,traintmp3,traintmp31,traintmp4,traintmp41

    return trainx,trainy,testx, testy

def plot1(ax,targets,yhat):
    plt.cla()
    ax.plot(targets.numpy(), color="orange")
    ax.plot(yhat.numpy(), 'g-', lw=3)
    plt.show()
    plt.pause(0.2)  # Note this correction
def plot_on_test(ax,targets,preds):
    plt.cla()
    flat_t = [item for sublist in targets for item in sublist]
    flat_p = [item for sublist in preds for item in sublist]
    ax.plot(flat_t, color="orange")
    ax.plot(flat_p, 'g-', lw=3)
    plt.show()
    plt.pause(0.2)  # Note this correction
    plt.savefig('evaluatePlot.png')
def plot_on_train(ax,targets,preds):
    plt.cla()
    flat_t = [item for sublist in targets for item in sublist]
    flat_p = [item for sublist in preds for item in sublist]
    ax.plot(flat_t, color="orange")
    ax.plot(flat_p, 'g-', lw=3)
    plt.show()
    plt.pause(0.2)  # Note this correction
    plt.savefig('traingPlot.png')

def plotloss(ax,trainlose,testlost):
    with open(trainlose) as f:
        trainlose = f.read().splitlines()
    with open(testlost) as f:
        testlose = f.read().splitlines()
    x=len(testlose)
    trainlose=[float(x) for x in trainlose]
    testlose = [float(x) for x in testlose]
    plt.cla()
    ax.plot(testlose, label='test lose')
    ax.plot(trainlose, label='train lose')
    plt.legend()
    plt.show()
    plt.pause(2)
    plt.savefig('trainAndTestLoseCurve.png')