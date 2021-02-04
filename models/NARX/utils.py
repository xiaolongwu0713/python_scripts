"""
Helper functions.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import numpy as np
import pandas as pd
import scipy.io

def read_data(args, debug=True):

    if args.feature=='fbands':
        train_prototype=args.traindataset
        test_prototype = args.testdataset
        #datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
        #datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'
        datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move' + str(train_prototype) + 'TrainData.mat'
        datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move' + str(test_prototype) + 'TestData.mat'
        raw1 = scipy.io.loadmat(datafile1)
        raw2 = scipy.io.loadmat(datafile2)
        train = raw1['train']  # (6299, 115)
        test = raw2['test']  # (2699, 115)
        tmp = np.concatenate((train, test), 0)  # (8998, 115)
        X = tmp[:, 0:-1]  # ([8998, 114])
        y = tmp[:, -1]  # ([8998])
    elif args.feature=='rawmove':
        train_prototype = args.traindataset
        test_prototype = args.testdataset
        datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move' + str(train_prototype) + '.mat'
    elif args.feature=='rawseeg':
        datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move1.mat'
    return X, y
