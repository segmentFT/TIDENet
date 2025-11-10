import math
import torch
import torch.nn as nn
from utility.layers import GraphConvolution
from utility.utils import normalize_A, generate_cheby_adj, ChannelwiseLayerNorm, ResBlock

class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super(DepthConv1d, self).__init__()

        self.skip = skip
        
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
          groups=hidden_channel,
          padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))


        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual
        

class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class ConvCrossAttention(nn.Module):
    def __init__(self, n_head, in_channels, kernel_size, dilation, dropout=0.1):
        super(ConvCrossAttention, self).__init__()

        self.n_head = n_head

        self.w_qs = DepthConv1d(in_channels, in_channels * 2, kernel_size, 
                                "same", dilation, False)
        self.w_ks = DepthConv1d(in_channels, in_channels * 2, kernel_size, 
                                "same", dilation, False)
        self.w_vs = DepthConv1d(in_channels, in_channels * 2, kernel_size, 
                                "same", dilation, False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.GroupNorm(1, in_channels, eps=1e-08)


    def forward(self, q, k, v):
        # q: [B, C, T]
        # k: [B, C, T]
        # v: [B, C, T]

        d_q, d_k, d_v = q.size(2), k.size(2), v.size(2)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        sz_b = q.size(0)

        residual = v

        q = self.w_qs(q)
        # q: [B, C, T]
        k = self.w_ks(k)
        # k: [B, C, T]
        v = self.w_vs(v)
        # v: [B, C, T]

        q = q.view(sz_b, len_q, self.n_head, d_q // self.n_head)
        # q: [B, C, H, T / H]
        k = k.view(sz_b, len_k, self.n_head, d_k // self.n_head)
        # k: [B, C, H, T / H]
        v = v.view(sz_b, len_v, self.n_head, d_v // self.n_head)
        # v: [B, C, H, T / H]

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q: [B, H, C, T / H]
        # k: [B, H, C, T / H]
        # v: [B, H, C, T / H]

        attn = torch.matmul(q / (d_k ** 0.5), k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, -1))
        # attn: [B, H, C, C]
        output = torch.matmul(attn, v)
        # output: [B, H, C, T / H]

        output = output.transpose(1, 2)
        # output: [B, C, H, T / H]
        output = output.reshape(sz_b, len_v, -1)
        # output: [B, C, T]
        output = output + residual

        output = self.layer_norm(output)

        return output

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, in_channels, kernel_size, dilation, n_head=1):
        super(MultiLayerCrossAttention, self).__init__()

        self.num_layers = num_layers

        self.projection = nn.Conv1d(in_channels * 4, in_channels, kernel_size, padding='same')
        self.output_layer_norm = nn.GroupNorm(1, in_channels, eps=1e-08)

        self.audio_encoder = nn.ModuleList()
        self.spike_encoder = nn.ModuleList()
        self.audio_layer_norm_list = nn.ModuleList()
        self.spike_layer_norm_list = nn.ModuleList()
        
        for i in range(num_layers):
            self.audio_layer_norm_list.append(nn.GroupNorm(1, in_channels, eps=1e-08))
            self.spike_layer_norm_list.append(nn.GroupNorm(1, in_channels, eps=1e-08))
        for i in range(num_layers):
            self.audio_encoder.append(ConvCrossAttention(n_head, in_channels, kernel_size, 
                                                         dilation))
            self.spike_encoder.append(ConvCrossAttention(n_head, in_channels, kernel_size, 
                                                         dilation))

    def forward(self, audio, spike):
        out_audio = audio
        out_spike = spike

        skip_audio = 0.
        skip_spike = 0.
        residual_audio = audio
        residual_spike = spike
        for i in range(self.num_layers):
            out_audio = self.audio_encoder[i](out_spike, out_audio, out_audio)
            out_spike = self.spike_encoder[i](out_audio, out_spike, out_spike)
            out_audio = out_audio + residual_audio
            out_audio = self.audio_layer_norm_list[i](out_audio)
            out_spike = out_spike + residual_spike
            out_spike = self.spike_layer_norm_list[i](out_spike)
            residual_audio = out_audio
            residual_spike = out_spike
            skip_audio += out_audio
            skip_spike += out_spike
        out = torch.cat((skip_audio, audio, out_spike, spike), dim=1)
        out = self.projection(out)
        out = self.output_layer_norm(out)
        return out        

class Dual_RNN_Block(nn.Module):
    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        # self.intra_norm = select_norm(norm, out_channels, 4)
        # self.inter_norm = select_norm(norm, out_channels, 4)
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class DPRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, CMCA_kernel=3, rnn_type='LSTM', norm='ln', K=250, 
                 dropout_rate=0.1, bidirectional=True, CMCA_layer_num=3, CMCA_n_head=1):
        super(DPRNN, self).__init__()
        
        self.K = K
        self.num_layers = num_layers

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, input_dim, 1e-8),
            nn.Conv1d(input_dim, hidden_dim, 1, bias=False)
        )
        self.fusion = MultiLayerCrossAttention(CMCA_layer_num, hidden_dim, CMCA_kernel, 
                                               1, CMCA_n_head)
        
        self.DPRNN= nn.ModuleList([])
        for _ in range(self.num_layers):
            self.DPRNN.append(Dual_RNN_Block(hidden_dim, hidden_dim, 
                                             rnn_type, norm, dropout_rate, bidirectional))
                    
        self.prelu = nn.PReLU()
        self.end_conv = nn.Conv1d(hidden_dim, output_dim, 1, bias=False)
        self.activation = nn.ReLU()    
        
    def forward(self, input, spike):
        # input: [B, 256, 7298]
        # spike: [B, 128, 7298]
        input = self.bottleneck(input)
        # input: [B, 128, 7298]  
        input = self.fusion(input, spike)
        # input: [B, 128, 7298]

        audio, gap = self._Segmentation(input, self.K)

        for i in range(self.num_layers):
            audio = self.DPRNN[i](audio)
        
        B, _, K, S = audio.shape
        audio = audio.view(B,-1, K, S)
        
        output = self._over_add(audio, gap)    
        output = self.prelu(output)
        output = self.end_conv(output)
        output = self.activation(output)

        return output
    
    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class AudioEncoder(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(AudioEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size, 
                      stride),
            nn.PReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size, 
                      stride),
            nn.PReLU()
        )

    def forward(self, x):
        # x: [B, 1, L]

        x1 = self.encoder1(x)
        # x1: [B, C, T]
        x2 = self.encoder2(x)
        # x2: [B, C, T]

        return x1, x2

class EEGEncoder(nn.Module):
    def __init__(self, num_electrodes, k_adj, enc_channel, feature_channel, kernel_size):
        super(EEGEncoder, self).__init__()

        class Chebynet(nn.Module):
            def __init__(self):
                super(Chebynet, self).__init__()

                self.K = k_adj
                self.gc = nn.ModuleList()
                for i in range(k_adj):
                    self.gc.append(GraphConvolution(num_electrodes, num_electrodes))

            def forward(self, x ,L):
        
                adj = generate_cheby_adj(L, self.K)

                for i in range(len(self.gc)):
                    if i == 0:
                        result = self.gc[i](x, adj[i])
                    else:
                        result += self.gc[i](x, adj[i])
                result = nn.functional.relu(result)

                return result
            
        class EncoderBranch(nn.Module):
            def __init__(self):
                super(EncoderBranch, self).__init__()

                self.batch_norm1 = nn.BatchNorm1d(256)
                self.batch_norm2 = nn.BatchNorm1d(29196)
                self.GCN_layer1 = Chebynet()
                self.GCN_layer2 = Chebynet()
                self.projection = nn.Conv1d(num_electrodes, feature_channel, kernel_size, 
                                            kernel_size // 2, bias=False)
                self.A1 = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
                self.A2 = nn.Parameter(torch.FloatTensor(num_electrodes , num_electrodes).cuda())
                nn.init.xavier_normal_(self.A1)
                nn.init.xavier_normal_(self.A2)

                self.up_sampler = nn.ConvTranspose1d(num_electrodes, num_electrodes, 381, 
                                                     113, groups=num_electrodes, bias=False)

                self.encoder = nn.Sequential(
                    ChannelwiseLayerNorm(feature_channel),
                    nn.Conv1d(feature_channel, feature_channel, 1),
                    ResBlock(feature_channel, feature_channel),
                    ResBlock(feature_channel, enc_channel),
                    ResBlock(enc_channel,enc_channel),
                    nn.Conv1d(enc_channel, feature_channel, 1),
                )

            def forward(self, spike):
                # spike: [B, 128, 256]

                spike = self.batch_norm1(spike.transpose(1, 2)).transpose(1, 2)
                spike = self.GCN_layer1(spike, normalize_A(self.A1))
                # spike: [B, 128, 256]

                spike = self.up_sampler(spike)
                # spike: [B, 128, 29196]

                spike = self.batch_norm2(spike.transpose(1, 2)).transpose(1, 2)
                spike = self.GCN_layer2(spike, normalize_A(self.A2))
                # spike: [B, 128, 29196]
        
                spike = self.projection(spike)
                spike = self.encoder(spike)
                # spike: [B, 64, 7298]

                return spike
            
        self.encoder_branch1 = EncoderBranch()
        self.encoder_branch2 = EncoderBranch()
        self.integration = nn.Sequential(
            nn.Conv1d(enc_channel, enc_channel, 1, bias=False),
            nn.GroupNorm(1, enc_channel, 1e-8),
            nn.PReLU()
        )

    def forward(self, x):
        # x: [B, 128, 256]

        x1 = self.encoder_branch1(x)
        # x1: [B, 64, 7298]
        x2 = self.encoder_branch2(x)
        # x2: [B, 64, 7298]
        x = self.integration(torch.concat([x1, x2], 1))
        # x: [B, 128, 7298]

        return x
