import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


class MPD(nn.Module):
    def __init__(self):
        super(MPD, self).__init__()

        self.periods = [2, 3, 5, 7, 11]
        self.sub_discriminators = nn.ModuleList()
        for i, period in enumerate(self.periods):
            self.sub_discriminators.append(SubMPD(period, norm = weight_norm if i != 0 else spectral_norm))


    def forward(self, audio, audio_pred):

        real_logits = []
        real_feature_maps = []
        fake_logits = []
        fake_feature_maps = []
        for d in self.sub_discriminators:
            real, real_feature_map = d(audio)
            real_logits.append(real)
            real_feature_maps.append(real_feature_map)


            fake, fake_feature_map = d(audio_pred)
            fake_logits.append(fake)
            fake_feature_maps.append(fake_feature_map)

        return real_logits, fake_logits, real_feature_maps, fake_feature_maps

    def remove_weight_norm(self):
        for disc in self.sub_discriminators:
            disc.remove_weight_norm()

class SubMPD(nn.Module):

    def __init__(self, period, norm):
        super(SubMPD, self).__init__()
        self.period = period
        self.norm = norm
        self.convs = nn.ModuleList()
        in_channels, out_channels = 1, 2 ** 5

        for _ in range(4):
            out_channels *= 2
            self.convs.append(self.norm(Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))))
            in_channels = out_channels
        
        self.convs.append(self.norm(Conv2d(out_channels, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))))
        self.convs.append(self.norm(Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))))


    def forward(self, x):
        # B x 1 x T -> B x 1 x T/p x p
        x = F.pad(x, pad=(0, self.period - x.shape[2] % self.period), mode="reflect")
        x = x.reshape((x.shape[0], 1, x.shape[2] // self.period, self.period))

        feature_map = []
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x), negative_slope=0.1)
            feature_map.append(x)

        x = self.convs[-1](x)
        feature_map.append(x)

        x = x.flatten(-2, -1)
        return x, feature_map
    
    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)

class MSD(nn.Module):
    def __init__(self):
        super(MSD, self).__init__()

        self.sub_discriminators = nn.ModuleList()
        for i in range(3):
            self.sub_discriminators.append(SubMSD(norm = weight_norm if i != 0 else spectral_norm))

        self.avgpool = AvgPool1d(4, 2, padding=2)

    def forward(self, audio, audio_pred):

        real_logits = []
        real_feature_maps = []
        fake_logits = []
        fake_feature_maps = []

        for i, d in enumerate(self.sub_discriminators):

            if i != 0:
                audio = self.avgpool(audio)
                audio_pred = self.avgpool(audio_pred)

            real, real_feature_map = d(audio)
            real_logits.append(real)
            real_feature_maps.append(real_feature_map)


            fake, fake_feature_map = d(audio_pred)
            fake_logits.append(fake)
            fake_feature_maps.append(fake_feature_map)

        return real_logits, fake_logits, real_feature_maps, fake_feature_maps

    def remove_weight_norm(self):
        for disc in self.sub_discriminators:
            disc.remove_weight_norm()

class SubMSD(nn.Module):

    def __init__(self, norm):
        super(SubMSD, self).__init__()
        self.norm = norm

        self.convs = nn.ModuleList([
            self.norm(Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
            self.norm(Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            self.norm(Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            self.norm(Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            self.norm(Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            self.norm(Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            self.norm(Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
        ])

    def forward(self, x):

        feature_map = []
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x), negative_slope=0.1)
            feature_map.append(x)

        x = self.convs[-1](x)
        feature_map.append(x)

        x = x.flatten(-2, -1)

        return x, feature_map

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)

