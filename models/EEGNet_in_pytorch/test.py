import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy
import math


Chans=64
Samples=128
dropoutRates=(0.25,0.25)
kernLength1=64
kernLength2=16
poolKern1=4
poolKern2=8
F1=4
D=2
F2=8
time_padding = int((kernLength1//2))

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0,dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation,dilation)

    h = math.floor((h_w[0] + 2*pad[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2*pad[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    return h, w

def get_model_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.data
    return params_dict

class deepwise_separable_conv(nn.Module):
    def __init__(self,nin,nout,kernelSize):
        super(deepwise_separable_conv,self).__init__()
        self.kernelSize = kernelSize
        self.time_padding = int(kernelSize//2)
        self.depthwise = nn.Conv2d(in_channels=nin,out_channels=nin,kernel_size=(1,kernelSize),padding=(0,time_padding),groups=nin,bias=False)
        self.pointwise = nn.Conv2d(in_channels=nin,out_channels=nout, kernel_size=1,groups=1,bias=False)
    def forward(self, input):
        dw = self.depthwise(input)
        pw = self.pointwise(dw)
        return pw
    def get_output_size(self,h_w):
        return convtransp_output_shape(h_w, kernel_size=(1,self.kernelSize), stride=1, pad=(0,time_padding), dilation=1)

# use below data to track the dimmension
data=torch.zeros(64,1,64,128)

output_sizes = {}
conv1 = nn.Conv2d(in_channels=1,out_channels=F1,kernel_size =(1,kernLength1),padding=(0,time_padding), stride=1,bias=False)
output_sizes['conv1']=convtransp_output_shape((Chans,Samples), kernel_size=(1,kernLength1), stride=1,
                                                   pad=(0,time_padding))
batchnorm1 = nn.BatchNorm2d(num_features=F1, affine=True)
depthwise1 = nn.Conv2d(in_channels=F1,out_channels=F1*D,kernel_size=(Chans,1),groups=F1,padding=0,bias=False)
output_sizes['depthwise1'] = convtransp_output_shape(output_sizes['conv1'], kernel_size=(Chans,1),stride=1, pad=0)
batchnorm2 = nn.BatchNorm2d(num_features=F1*D, affine=True)
activation_block1 = nn.ELU()
# avg_pool_block1 = nn.AvgPool2d((1,poolKern1))
# output_sizes['avg_pool_block1'] = convtransp_output_shape(output_sizes['depthwise1'], kernel_size=(1, poolKern1),
#                                                           stride=(1,poolKern1), pad=0)
avg_pool_block1 = nn.AdaptiveAvgPool2d((1,int(output_sizes['depthwise1'][1]/4)))
output_sizes['avg_pool_block1'] = (1,int(output_sizes['depthwise1'][1]/4))
dropout_block1 = nn.Dropout(p=dropoutRates[0])

#block2
separable_block2 = deepwise_separable_conv(nin=F1*D,nout=F2,kernelSize=kernLength2)
output_sizes['separable_block2'] = separable_block2.get_output_size(output_sizes['avg_pool_block1'])
activation_block2 = nn.ELU()
# avg_pool_block2 = nn.AvgPool2d((1,poolKern2))
# output_sizes['avg_pool_block2'] = convtransp_output_shape(output_sizes['separable_block2'],
#                                                                kernel_size=(1, poolKern2),
#                                                                stride=(1, poolKern2), pad=0)
avg_pool_block2 = nn.AdaptiveAvgPool2d((1,int(output_sizes['separable_block2'][1]/4)))
output_sizes['avg_pool_block2'] = (1,int(output_sizes['separable_block2'][1]/4))

dropout_block2 = nn.Dropout(dropoutRates[1])

flatten = nn.Flatten()
n_size = get_features_dim(Chans,Samples)
dense = nn.Linear(n_size,nb_classes)

def get_features_dim(self,Chans,Samples):
    bs = 1
    x = Variable(torch.rand((bs,1,Chans, Samples)))
    output_feat,out_dims = forward_features(x)
    n_size = output_feat.data.view(bs, -1).size(1)
    return n_size