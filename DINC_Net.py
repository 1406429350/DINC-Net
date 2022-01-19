'''
       Paper: Cardiopulmonary Auscultation Enhancement with a Two-Stage Noise Cancellation Approach
       The first version of DIDN's network architecture
'''

import torch
import torch.nn as nn


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input.
       It is also known as a fractionally-strided convolution
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=128, out_channels=256,
                 kernel_size=3, dilation=1, norm='gln', causal=False, skip_con=True):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.Output = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.skip_con = skip_con

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        c = self.PReLU_2(c)
        c = self.norm_2(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        if self.skip_con:
            Sc = self.Sc_conv(c)
            c = self.Output(c)
            return Sc, c+x
        c = self.Output(c)
        return x+c


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
           this module has learnable per-element affine parameters
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.sequential = nn.Sequential(
            Conv1D(in_channels, out_channels, kernel_size, stride=stride),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            Conv1D(out_channels, out_channels, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU()
        )

    def forward(self, x):
        '''
           x: [B, T]
           out: [B, N, T]
        '''
        x = self.sequential(x)
        return x


class Decoder(nn.Module):
    '''
        Decoder
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''
    def __init__(self, N, kernel_size=16, stride=16 // 2):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.PReLU(),
            nn.ConvTranspose1d(N, N, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose1d(N, 1, kernel_size=kernel_size, stride=stride, bias=True)
        )

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        x = self.sequential(x)
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class Interaction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Interaction, self).__init__()

        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.conv1x1 = Conv1D(in_channels, in_channels, 1)
        self.norm_1 = select_norm(norm, out_channels)
        self.norm_2 = select_norm(norm, in_channels)
        self.conv_2 = Conv1D(out_channels, in_channels, kernel_size=3, stride=1,  padding=1)
        self.dwconv = Conv1D(in_channels, in_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)

    def forward(self, x1, x2, x3):
        x11 = self.norm_2(x1)   # 128
        x11 = self.conv1x1(x11)   # 128
        x22 = self.norm_2(x2)
        x22 = self.conv1x1(x22)   # 128
        x = torch.cat((x11, x22), dim=1)   # 256
        x = self.norm_1(x)
        x = self.conv_2(x)   # 128
        x = self.PReLU_1(x)
        x = self.norm_2(x)
        x = x*x3   # 256
        return x+x3


class Separation(nn.Module):
    '''
       R	Number of repeats
       X	Number of convolutional blocks in each repeat
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       norm The type of normalization(gln, cl, bn)
       causal  Two choice(causal or noncausal)
       skip_con Whether to use skip connection
    '''
    # B 128 H 256
    def __init__(self, R, X, B, H, P, norm='gln', causal=False, skip_con=True):
        super(Separation, self).__init__()
        self.separation = nn.ModuleList([])
        for x in range(X):
            self.separation.append(Conv1D_Block(
                B, H, P, 2**x, norm, causal, skip_con))
        self.skip_con = skip_con

    def forward(self, x):
        '''
           x: [B, N, L]
           out: [B, N, L]
        '''
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.separation)):
                skip, out = self.separation[i](x)
                skip_connection = skip_connection + skip
                x = out
            return x, skip_connection
        else:
            for i in range(len(self.separation)):
                out = self.separation[i](x)
                x = out
            return x


class DINC_Net(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=128,
                 L=16,
                 B=128,
                 H=256,
                 P=3,
                 X=8,
                 R=4,
                 norm="gln",
                 num_spks=2,
                 activate="sigmoid",
                 causal=False,
                 skip_con=True):
        super(DINC_Net, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Encoder(1, N, L, stride=L // 2)
        self.BottleN_S = Conv1D(N, B, 1)
        self.LayerN_S = select_norm('gln', N)
        # n x B x T => n x B x T
        self.separation_0 = Separation(R, X, B, H, P, norm=norm, causal=causal, skip_con=skip_con)

        self.interaction_0 = Interaction(B, H)

        self.separation_1 = Separation(R, X, B, H, P, norm=norm, causal=causal, skip_con=skip_con)

        self.interaction_1 = Interaction(B, H)

        self.separation_2 = Separation(R, X, B, H, P, norm=norm, causal=causal, skip_con=skip_con)

        self.interaction_2 = Interaction(B, H)

        self.separation_3 = Separation(R, X, B, H, P, norm=norm, causal=causal, skip_con=skip_con)

        self.interaction_3 = Interaction(B, H)

        self.PReLU_1 = nn.PReLU()

        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, N, 1)

        # n x N x T => n x 1 x L
        self.decoder = Decoder(N, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_spks = num_spks

    def forward(self, x):
        # x: n x 1 x L => n x N x T
        w1 = self.encoder(x[0])    # 128
        w2 = self.encoder(x[1])   # 128

        # Interactive Noise Cancellation Module
        # n x N x L => n x B x L
        e1 = self.interaction_0(w1, w2, w1)
        w3, w41 = self.separation_0(e1)
        w3 = self.interaction_1(w1, w2, w3)
        w3, w42 = self.separation_1(w3)
        w3 = self.interaction_2(w1, w2, w3)
        w3, w43 = self.separation_2(w3)
        w3 = self.interaction_3(w1, w2, w3)
        w3, w44 = self.separation_3(w3)
        m1 = self.PReLU_1(w41+w42+w43+w44)
        m1 = self.gen_masks(m1)
        m1 = self.LayerN_S(m1)
        m1 = self.activation(m1)
        d = w1*m1

        s = self.decoder(d)
        return s


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_DINC_Net():
    x = torch.randn(1, 32000)
    y = torch.randn(1, 32000)
    nnet = DINC_Net()
    s = nnet([x, y])
    print(str(check_parameters(nnet))+' Mb')
    print(nnet)


if __name__ == "__main__":
    test_DINC_Net()
