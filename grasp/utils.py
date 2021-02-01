from config import *
import os
import numpy as np
import scipy.io
import re

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

def


