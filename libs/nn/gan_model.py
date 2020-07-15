import torch.nn as nn
import torch.nn.functional as F
from libs.utils.spectral_norm import spectral_norm#import SpectralNorm as spectral_norm

class Interpolate2D(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Interpolate2D, self).__init__()
        self.sf = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.sf, mode=self.mode)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidulBlock(nn.Module):
    def __init__(self, inc, outc, sample='none'):
        super(ResidulBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(inc, outc, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(outc, outc, 3, 1, 1))
        self.conv_sc = spectral_norm(nn.Conv2d(inc, outc, 1, 1, 0)) if inc != outc else False

        if sample == 'up':
            self.sample = Interpolate2D(scale_factor=2)
        else:
            self.sample = None

        self.bn1 = nn.BatchNorm2d(inc)
        self.bn2 = nn.BatchNorm2d(outc)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.act(self.bn1(x))

        if self.sample:
            h = self.sample(h)
            x = self.sample(x)

        h = self.conv1(h)
        h = self.act(self.bn2(h))
        h = self.conv2(h)

        if self.conv_sc:
            x = self.conv_sc(x)
        return x + h


# generator
class Generator(nn.Module):
    def __init__(self, ch_input, use_decode):
        super(Generator, self).__init__()
        ch_output = 3
        base_dim = 64
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_input, base_dim * 1, 3, 1, 1)),
            nn.BatchNorm2d(base_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(base_dim * 1, base_dim * 2, 3, 1, 1)),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1)),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

        if use_decode:
            self.decoder = nn.Sequential(
                spectral_norm(nn.Conv2d(base_dim * 4, base_dim * 2, 3, 1, 1)),
                ResidulBlock(base_dim * 2, base_dim * 2),
                ResidulBlock(base_dim * 2, base_dim * 2),
                ResidulBlock(base_dim * 2, base_dim * 2),
                ResidulBlock(base_dim * 2, base_dim * 2),
                ResidulBlock(base_dim * 2, base_dim * 2, sample='up'),
                ResidulBlock(base_dim * 2, base_dim * 1, sample='up'),
                ResidulBlock(base_dim * 1, base_dim * 1, sample='up'),
                nn.BatchNorm2d(base_dim * 1),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(base_dim * 1, ch_output, 3, 1, 1)),
                nn.Sigmoid(),
            )
        else:
            self.decoder = lambda x: x

    def encode(self, input):
        return self.encoder(input)

    def decode(self, h):
        return self.decoder(h)


# discriminator
class Discriminator(nn.Module):
    def __init__(self, ch_input):
        super(Discriminator, self).__init__()
        base_dim = 64
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_input, base_dim * 1, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 1, base_dim * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 2, base_dim * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 4, base_dim * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 8, 1, 3, 1, 1)),
        )

    def forward(self, x):
        return self.net(x)
