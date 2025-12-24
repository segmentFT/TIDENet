import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class VoiceEncoder(nn.Module):
    def __init__(self, out_channels=128, kernel_size=8, stride=4):
        super(VoiceEncoder, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size, stride=stride),
            nn.PReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size, stride=stride),
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
    def __init__(self, num_electrodes=128, num_adjacents=3, out_channels=128, kernel_size=8):
        super(EEGEncoder, self).__init__()

        class Chebynet(nn.Module):
            def __init__(self):
                super(Chebynet, self).__init__()

                class GraphConvolution(nn.Module):
                    def __init__(self):
                        super(GraphConvolution, self).__init__()

                        self.weight = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
                        nn.init.xavier_normal_(self.weight)

                    def forward(self, x, adjacent_matrix):
                        # x: [B, Ce, Le]
                        # adjacent_matrix: [Ce, Ce]

                        output = torch.matmul(self.weight, torch.matmul(adjacent_matrix, x))
                        # output: [B, Ce, Le]
                        
                        return output

                self.graph_convs = nn.ModuleList([GraphConvolution() for _ in range(num_adjacents)])

            def generate_cheby_adj(self, laplacian_matrix):
                # laplacian_matrix: [Ce, Ce]

                support = []
                for i in range(num_adjacents):
                    if i == 0:
                        support.append(torch.eye(laplacian_matrix.size(-1)).cuda())
                    elif i == 1:
                        support.append(laplacian_matrix)
                    else:
                        temp = torch.matmul(2*laplacian_matrix, support[-1]) - support[-2]
                        support.append(temp)
                
                return support

            def forward(self, x, laplacian_matrix):
                # x: [B, Ce, Le]
                # laplacian_matrix: [Ce, Ce]
        
                adjacent_matrices = self.generate_cheby_adj(laplacian_matrix)

                for i in range(num_adjacents):
                    if i == 0:
                        result = self.graph_convs[i](x, adjacent_matrices[i])
                    else:
                        result = result + self.graph_convs[i](x, adjacent_matrices[i])
                
                result = F.relu(result)

                return result
            
        class EncoderBranch(nn.Module):
            def __init__(self):
                super(EncoderBranch, self).__init__()

                class ResBlock(nn.Module):
                    def __init__(self, in_dims, out_dims):
                        super(ResBlock, self).__init__()

                        self.operations = nn.Sequential(
                            nn.Conv1d(in_dims, out_dims, 3, padding=1, bias=False),
                            nn.BatchNorm1d(out_dims),
                            nn.PReLU(),
                            nn.Conv1d(out_dims, out_dims, 3, padding=1, bias=False),
                            nn.BatchNorm1d(out_dims)
                        )
                        self.prelu = nn.PReLU()

                        if in_dims != out_dims:
                            self.downsample = True
                            self.conv_downsample = nn.Conv1d(in_dims, out_dims, 1, bias=False)
                        else:
                            self.downsample = False

                    def forward(self, x):
                        # x: [B, C1, T]

                        y = self.operations(x)
                        # y: [B, C2, T]

                        if self.downsample:
                            y += self.conv_downsample(x)
                        else:
                            y += x
                        
                        return self.prelu(y)

                self.batch_norm1 = nn.BatchNorm1d(256)
                self.batch_norm2 = nn.BatchNorm1d(29184)
                self.GCN_layer1 = Chebynet()
                self.GCN_layer2 = Chebynet()
                
                self.A1 = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
                self.A2 = nn.Parameter(torch.FloatTensor(num_electrodes , num_electrodes).cuda())
                nn.init.xavier_normal_(self.A1)
                nn.init.xavier_normal_(self.A2)

                self.up_sampler = nn.ConvTranspose1d(num_electrodes, num_electrodes, 369, 
                                                     113, groups=num_electrodes, bias=False)

                self.projection = nn.Conv1d(num_electrodes, out_channels//2, kernel_size, 
                                            kernel_size//2, bias=False)
                self.layer_norm = nn.LayerNorm(out_channels//2)
                self.encoder = nn.Sequential(
                    nn.Conv1d(out_channels//2, out_channels//2, 1),
                    ResBlock(out_channels//2, out_channels//2),
                    ResBlock(out_channels//2, out_channels),
                    ResBlock(out_channels,out_channels),
                    nn.Conv1d(out_channels, out_channels//2, 1),
                )

            def normalize_A(self, A):
                # A: [Ce, Ce]

                ones = torch.ones(num_electrodes, num_electrodes).cuda()
                # ones: [Ce, Ce]
                diag = torch.eye(num_electrodes, num_electrodes).cuda()
                # diag: [Ce, Ce]

                A = F.relu(A)
                A = A * (ones - diag)
                A = A + A.T

                degree_matrix = A.sum(-1)
                # degree_matrix: [Ce]
                degree_matrix = 1 / torch.sqrt((degree_matrix + 1e-10))
                # degree_matrix: [Ce]
                degree_matrix = torch.diag_embed(degree_matrix)
                # degree_matrix: [Ce, Ce]
                laplacian_matrix = diag - torch.matmul(torch.matmul(degree_matrix, A), degree_matrix)
                # laplacian_matrix: [Ce, Ce]
                L_norm = laplacian_matrix - diag
                # L_norm: [Ce, Ce]

                return L_norm

            def forward(self, x):
                # spike: [B, C3, 256]

                x = torch.transpose(x, -2, -1)
                # x: [B, 256, Ce]
                x = self.batch_norm1(x)
                # x: [B, 256, Ce]
                x = torch.transpose(x, -2, -1)
                # x: [B, Ce, 256]
                x = self.GCN_layer1(x, self.normalize_A(self.A1))
                # x: [B, Ce, 256]

                x = self.up_sampler(x)
                # x: [B, Ce, 29184]

                x = torch.transpose(x, -2, -1)
                # x: [B, 29184, Ce]
                x = self.batch_norm2(x)
                # x: [B, Ce, 29184]
                x = torch.transpose(x, -2, -1)
                # x: [B, 29184, Ce]
                x = self.GCN_layer2(x, self.normalize_A(self.A2))
                # spike: [B, Ce, 29184]
        
                x = self.projection(x)
                # x: [B, C/2, T]
                x = torch.transpose(x, -2, -1)
                # x: [B, T, C/2]
                x = self.layer_norm(x)
                # x: [B, T, C/2]
                x = torch.transpose(x, -2, -1)
                # x: [B, C/2, T]
                x = self.encoder(x)
                # x: [B, C/2, T]

                return x
            
        self.encoder_branch1 = EncoderBranch()
        self.encoder_branch2 = EncoderBranch()
        self.integration = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels, 1e-8),
            nn.PReLU()
        )

    def forward(self, x):
        # x: [B, 128, 256]

        x1 = self.encoder_branch1(x)
        # x1: [B, C/2, T]
        x2 = self.encoder_branch2(x)
        # x2: [B, C/2, T]
        x = self.integration(torch.concat([x1, x2], 1))
        # x: [B, C, T]

        return x


class Separator(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, rnn_type='LSTM', K=250, num_layers={"main": 4, "fusion": 3}, ):
        super(Separator, self).__init__()
        
        self.K = K
        self.num_layers = num_layers

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, in_channels, 1e-8),
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        self.fusion = MultiLayerCrossAttention(num_layers["fusion"], out_channels=out_channels)

        self.DPRNN= nn.ModuleList([
            Dual_RNN_Block(out_channels, rnn_type) for _ in range(self.num_layers["main"])])
        
        self.output_layer = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
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


class TIDENet(nn.Module):
    def __init__(self, num_electrodes=128, num_adjacents=3, out_channels=128, kernel_size=8, num_layers={"main": 4, "fusion": 3}, 
                 rnn_type='LSTM', K=250):
        super(TIDENet, self).__init__()

        self.EEG_encoder = EEGEncoder(num_electrodes, num_adjacents, out_channels, 
                                        kernel_size)
        self.voice_encoder = VoiceEncoder(out_channels, kernel_size, kernel_size//2)

        self.projection = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels)
        )
        
        self.mask_net = Separator(out_channels*2, out_channels, rnn_type, K, 
                                  num_layers)

        self.decoder = nn.ConvTranspose1d(out_channels, 1, kernel_size, 
                                          kernel_size // 2, bias=False)
        
    def forward(self, voice, raw_EEG):
        # voice: [B, 1, 29184]
        # raw_EEG: [B, Ce, 256]

        raw_EEG = self.EEG_encoder(raw_EEG)
        # spike_input: [B, C, T]
        voice1, voice2 = self.voice_encoder(voice)
        # voice1: [B, C, T]
        # voice2: [B, C, T]

        voice = torch.concat([voice1, voice2], 1)
        # voice: [B, C*2, T]
        mask = F.sigmoid(self.mask_net(voice, raw_EEG))
        # mask: [B, C, T]

        voice1 = torch.transpose(voice1, -2, -1)
        # voice1: [B, T, C]
        voice1 = self.projection(voice1)
        # voice1: [B, T, C]
        voice1 = torch.transpose(voice1, -2, -1)
        # voice1: [B, C, T]

        mask = mask.transpose(-2, -1)
        # mask: [B, T, C]
        mask = self.projection(mask)
        # mask: [B, T, C]
        mask = torch.transpose(mask, -2, -1)
        # mask: [B, C, T]
        
        output = self.decoder(voice1 * mask)
        # output: [B, 1, 29184]
        
        return output


def test():
    x = torch.randn(2, 1, 29184).cuda()
    y = torch.randn(2, 128, 256).cuda()
    net = TIDENet().cuda()

    z = net(x, y)
    print(z.shape)

    for name, param in net.named_parameters():
        print(f"{name}: {param.shape}")

    params = filter(lambda p: p.requires_grad, net.parameters())
    num_params = np.sum([np.prod(p.shape) for p in params]) / 1e6
    print(num_params)

if __name__ == "__main__":
    test()
