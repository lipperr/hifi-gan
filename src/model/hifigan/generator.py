import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm


class Generator(nn.Module):

    def __init__(self, hidden_dim=128, ks_convT=[16, 16, 4, 4], ks_resblocks=[3, 7, 11], d_resblocks=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super(Generator, self).__init__()
        self.conv1 = weight_norm(Conv1d(in_channels=80, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3))

        self.upsample_convT = nn.ModuleList()
        for i, k in enumerate(ks_convT):
            self.upsample_convT.append(weight_norm(
                ConvTranspose1d(hidden_dim // (2 ** i), hidden_dim // (2 ** (i + 1)),
                                kernel_size=k, stride=k//2, padding= (k - k//2) // 2)))

        self.mrf_blocks = nn.ModuleList()
        for i in range(len(self.upsample_convT)):
            mrf_channels = hidden_dim //(2 ** (i + 1))
            self.mrf_blocks.append(MRF(mrf_channels, ks_resblocks, d_resblocks))

        self.conv2 = weight_norm(Conv1d(in_channels=mrf_channels, out_channels=1, kernel_size=7, stride=1, padding=3))

    def forward(self, melspectrogram, **batch):

        x = self.conv1(melspectrogram)
        for i in range(len(self.upsample_convT)):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.upsample_convT[i](x)
            x = self.mrf_blocks[i](x)

        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = torch.tanh(x)

        return x
    
    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)
        for conv in self.upsample_convT:
            remove_weight_norm(conv)

class MRF(nn.Module):

    def __init__(self, channels, kernel_sizes, dilations):
        super(MRF, self).__init__()

        self.resblocks = nn.ModuleList()
        for (k, d) in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock(channels, k, d))

    def forward(self, x):
        x = self.resblocks[0](x)
        for i in range(1, len(self.resblocks)):
            x = x + self.resblocks[i](x)
        x = x / len(self.resblocks)
        return x
    
    def remove_weight_norm(self):
        for block in self.resblocks:
            block.remove_weight_norm()


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()

        self.conv_layer1 = nn.ModuleList()
        for i in range(3):
            self.conv_layer1.append(weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                                                        padding=(kernel_size*dilation[i] - dilation[i])//2,
                                                        dilation=dilation[i])))
        self.conv_layer2 = nn.ModuleList()
        for i in range(3):
            self.conv_layer2.append(weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                                                        padding=(kernel_size - 1)//2,
                                                        dilation=1)))
            
    def forward(self, x):

        for i in range(3):
            x_res = F.leaky_relu(x, negative_slope=0.1)
            x_res = self.conv_layer1[i](x_res)
            x_res = F.leaky_relu(x_res, negative_slope=0.1)
            x_res = self.conv_layer2[i](x_res)
            x = x + x_res
        return x
    
    def remove_weight_norm(self):

        for i in range(3):
            remove_weight_norm(self.conv_layer1[i])
            remove_weight_norm(self.conv_layer2[i])
