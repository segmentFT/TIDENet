import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_A(A,lmax=2):   # A(128,128)
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm    #(128,128)


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)
    return support

        
class ChannelwiseLayerNorm(nn.LayerNorm):
    """
    Channel-wise layer normalization based on nn.LayerNorm
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, *args, **kwargs):
        super(ChannelwiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(
                self.__name__))
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return y
