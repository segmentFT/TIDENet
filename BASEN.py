import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utility import models
from util import print_size


class BASEN(nn.Module):
    def __init__(self, num_electrodes=128, k_adj=3, enc_channel=128, feature_channel=64, kernel_size=8, num_layers=4, rnn_type='LSTM', 
                 norm='ln', K=250, dropout_rate=0.1, bidirectional=True, CMCA_kernel=3, CMCA_layer_num=3, CMCA_n_head=1):
        super(BASEN, self).__init__()

        self.win_len = kernel_size
        self.stride = self.win_len // 2

        self.spike_encoder =models.EEGEncoder(num_electrodes, k_adj, enc_channel, 
                                              feature_channel, kernel_size)
        self.speech_encoder = models.VoiceEncoder(enc_channel, kernel_size, kernel_size // 2)
        self.projection = nn.Conv1d(enc_channel, enc_channel, 1)
        
        self.DPRNN = models.Separator(enc_channel * 2, enc_channel, enc_channel, 
                                  num_layers, CMCA_kernel, rnn_type, norm, K, 
                                  dropout_rate, bidirectional, CMCA_layer_num, 
                                  CMCA_n_head)

        self.decoder = nn.ConvTranspose1d(enc_channel, 1, kernel_size, 
                                          kernel_size // 2, bias=False)

    def pad_signal(self, input):
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        n_ch = input.size(1)
        nsample = input.size(2)
        rest = self.win_len - (self.stride + nsample % self.win_len) % self.win_len
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, n_ch, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, n_ch, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, speech_input, spike_input):
        pass
    
    