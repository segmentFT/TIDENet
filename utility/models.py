import math
import torch
import torch.nn as nn
from utility.layers import GraphConvolution
from utility.utils import normalize_A, generate_cheby_adj, ChannelwiseLayerNorm, ResBlock


def _padding(input, K):
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

def _Segmentation(input, K):
    '''
        the segmentation stage splits
        K: chunks of length
        P: hop size
        input: [B, N, L]
        output: [B, N, K, S]
    '''
    B, N, L = input.shape
    P = K // 2
    input, gap = _padding(input, K)
    # [B, N, K, S]
    input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
    input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
    input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

    return input.contiguous(), gap\
    
def _over_add(input, gap):
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


class DepthConv1d(nn.Module):

    def __init__(self, out_channels, hidden_channels):
        super(DepthConv1d, self).__init__()

        self.operations = nn.Sequential(
            nn.Conv1d(out_channels, hidden_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels, 1e-8),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels, 1e-8),
            nn.Conv1d(hidden_channels, out_channels, 1)
        )

    def forward(self, x):
        # x: [B, C, T]
        
        return self.operations(x)
        

class ConvCrossAttention(nn.Module):
    def __init__(self, out_channels):
        super(ConvCrossAttention, self).__init__()

        self.to_q = DepthConv1d(out_channels, out_channels*2)
        self.to_k = DepthConv1d(out_channels, out_channels*2)
        self.to_v = DepthConv1d(out_channels, out_channels*2)
        self.norm = nn.GroupNorm(1, out_channels, 1e-8)

    def forward(self, x, y):
        # x: [B, C, T]
        # y: [B, C, T]

        q = self.to_q(x)
        # q: [B, C, T]
        k = self.to_k(y)
        # k: [B, C, T]
        v = self.to_v(y)
        # v: [B, C, T]

        k = torch.transpose(k, -2, -1)
        # k: [B, T, C]
        p_attn = torch.matmul(q, k) / math.sqrt(k.size(1))
        # p_attn: [B, C, C]
        attn = torch.dropout(p_attn.softmax(-1), 0.1, self.training)
        # attn:  [B, C, C]
        output = torch.matmul(attn, v)
        # output: [B, C, T]

        return self.norm(output + y)

class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, out_channels):
        super(MultiLayerCrossAttention, self).__init__()

        self.num_layers = num_layers

        self.voice_attns = nn.ModuleList([ConvCrossAttention(out_channels) for _ in range(num_layers)])
        self.EEG_attns = nn.ModuleList([ConvCrossAttention(out_channels) for _ in range(num_layers)])
        self.voice_norms = nn.ModuleList([
            nn.GroupNorm(1, out_channels, 1e-8) for _ in range(num_layers)])
        self.EEG_norms = nn.ModuleList([
            nn.GroupNorm(1, out_channels, 1e-8) for _ in range(num_layers)])
        
        self.projection = nn.Sequential(
            nn.Conv1d(out_channels*4, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels, 1e-8)
        )

    def forward(self, voice, EEG_cue):
        # voice: [B, C, T]
        # EEG_cue: [B, C, T]

        residual_cue = torch.clone(EEG_cue)
        residual_voice = torch.clone(voice)

        for i in range(self.num_layers):
            mid_feat = self.voice_attns[i](EEG_cue, voice)
            # mid_feat: [B, C, T]
            EEG_cue = self.EEG_norms[i](self.EEG_attns[i](voice, EEG_cue) + EEG_cue)
            # EEG_cue: [B, C, T]
            voice = self.voice_norms[i](mid_feat + voice)
            # voice: [B, C, T]

            if i == 0:
                skip_voice = voice
                skip_cue = EEG_cue
            else:
                skip_voice = skip_voice + voice
                skip_cue = skip_cue + EEG_cue

        output = torch.concat([skip_voice, residual_voice, EEG_cue, residual_cue], 1)
        # output: [B, C*4, T]
        output = self.projection(output)
        # output: [B, C, T]

        return output       

class Dual_RNN_Block(nn.Module):
    def __init__(self, out_channels, rnn_type='LSTM'):
        super(Dual_RNN_Block, self).__init__()

        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, out_channels, 1, batch_first=True, dropout=0.1, bidirectional=True)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, out_channels, 1, batch_first=True, dropout=0.1, bidirectional=True)
        
        # Normalization
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

        # Linear
        self.intra_linear = nn.Linear(out_channels*2, out_channels)
        self.inter_linear = nn.Linear(out_channels*2, out_channels)
    
    def forward(self, x):
        # x: [B, C, K, S]

        B, C, K, S = x.shape

        # intra RNN
        
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, C)
        # [B*S, K, C]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [B*S, K, H]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B*S, K, C]
        intra_rnn = intra_rnn.view(B, S, K, C)
        # [B, S, K, C]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        # [B, C, K, S]
        intra_rnn = self.intra_norm(intra_rnn)
        # [B, C, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, C)
        # [B*K, S, C]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [B*K, S, C*2]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B*K, S, C]
        inter_rnn = inter_rnn.view(B, K, S, C)
        # [B, K, S, C]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        # [B, C, K, S]
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, C, K, S]
        out = inter_rnn + intra_rnn

        return out


class DPRNN(nn.Module):
    def __init__(self, out_channels=128, hidden_channels=128, num_layers={"main": 4, "fusion": 3}, rnn_type='LSTM', K=250):
        super(DPRNN, self).__init__()
        
        self.K = K
        self.num_layers = num_layers

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, out_channels*2, 1e-8),
            nn.Conv1d(out_channels*2, hidden_channels, 1, bias=False)
        )
        self.fusion = MultiLayerCrossAttention(num_layers["fusion"], out_channels=hidden_channels)
        self.DPRNN= nn.ModuleList([
            Dual_RNN_Block(hidden_channels, rnn_type) for _ in range(self.num_layers)])
        self.output_layer = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(hidden_channels, out_channels, 1, bias=False),
            nn.PReLU()
        )
        
    def forward(self, voice, EEG_cue):
        # voice: [B, C*2, T]
        # EEG_cue: [B, C, T]

        voice = self.bottleneck(voice)
        # voice: [B, C, T]  
        voice = self.fusion(voice, EEG_cue)
        # input: [B, C, T]

        voice, gap = _Segmentation(voice, self.K)
        # voice: [B, C, K, S]

        for i in range(self.num_layers["main"]):
            voice = self.DPRNN[i](voice)
        
        output = _over_add(voice, gap)
        # output: [B, C, T]
        output = self.output_layer(output)
        # output: [B, C, T]

        return output


class AudioEncoder(nn.Module):
    def __init__(self, out_channels=128, kernel_size=8, stride=4):
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
