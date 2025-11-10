import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utility.utils import ChannelwiseLayerNorm
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
        self.speech_encoder = models.AudioEncoder(enc_channel, kernel_size, kernel_size // 2)
        self.layer_norm = ChannelwiseLayerNorm(enc_channel)
        self.projection = nn.Conv1d(enc_channel, enc_channel, 1)
        
        self.DPRNN = models.DPRNN(enc_channel * 2, enc_channel, enc_channel, 
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
        # speech_input: [B, 1, 29184]
        # spike_input: [B, 128, 256]

        speech_input, rest = self.pad_signal(speech_input)
        batch_size = speech_input.size(0)
        spike_input = self.spike_encoder(spike_input)
        # spike_input: [B, 128, 7298]
        
        speech_len = speech_input.size(2)
        x1, x2 = self.speech_encoder(speech_input)
        # x1: [B, 128, 7298]
        # x2: [B, 128, 7298]
        speech_output = torch.sigmoid(self.DPRNN(torch.concat([x1, x2], 1), spike_input))

        enc_output = self.layer_norm(x1)
        enc_output = self.projection(enc_output)
        
        masks = self.layer_norm(speech_output)
        masks = self.projection(masks)
        masked_output = enc_output * masks
        
        output = self.decoder(masked_output)
        output = F.pad(output, (0, speech_len - output.size(2)), "constant", 0)
        output = output[:, :, self.stride: -(rest + self.stride)].contiguous()
        output = output.view(batch_size, 1, -1)
        
        return output
    

if __name__ == "__main__":
    x = torch.rand(2, 1, 29184).cuda()
    y = torch.rand(2, 128, 256).cuda()
    net = BASEN().cuda()

    z = net(x, y)
    print(z.shape)
    print_size(net)
    print_size(net.DPRNN)
    print_size(net.DPRNN.fusion)
    print_size(net.DPRNN.fusion.projection)
    print_size(net.DPRNN.fusion.audio_encoder)
    print_size(net.DPRNN.fusion.spike_encoder)
    print_size(net.DPRNN.fusion.audio_encoder[0])
    print_size(net.DPRNN.fusion.audio_encoder[0].w_qs)
    