import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


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

class EncoderS(nn.Module):
    def __init__(self):
        super(EncoderS, self).__init__()
        base_dim = 16
        ch_style = 1
        self.hook_outputs = []

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_style, base_dim * 1, 3, 1, 1)),
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

        # add hook
        for idx in [3, 7]:
            self.model[idx].register_forward_hook(self.hook)

    def hook(self, model, input, output):
        self.hook_outputs += [output]

    def forward(self, image):
        self.hook_outputs = []

        last_feature = self.model(image)
        b, c, h, w = last_feature.size()

        # aggregate
        previous_features = [F.interpolate(feature, size=(h,w)) for feature in self.hook_outputs]
        last_feature = torch.cat(previous_features + [last_feature], dim=1)

        return last_feature

class EncoderR(nn.Module):
    def __init__(self):
        super(EncoderR, self).__init__()
        base_dim = 16
        ch_style = 3
        self.hook_outputs = []

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_style, base_dim * 1, 3, 1, 1)),
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

        # add hook
        for idx in [3, 7]:
            self.model[idx].register_forward_hook(self.hook)

    def hook(self, model, input, output):
        self.hook_outputs += [output]

    def forward(self, image):
        self.hook_outputs = []

        last_feature = self.model(image)
        b, c, h, w = last_feature.size()

        # aggregate
        previous_features = [F.interpolate(feature, size=(h, w)) for feature in self.hook_outputs]
        last_feature = torch.cat(previous_features + [last_feature], dim=1)

        return last_feature


class DecoderS(nn.Module):
    def __init__(self):
        super(DecoderS, self).__init__()
        base_dim = 16
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(112, base_dim * 4, 3, 1, 1)),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 2, sample='up'),
            ResidulBlock(base_dim * 2, base_dim * 1, sample='up'),
            ResidulBlock(base_dim * 1, base_dim * 1, sample='up'),
            nn.BatchNorm2d(base_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 1, 3, 3, 1, 1)),
            nn.Tanh(),
        )

    def forward(self, h):
        return self.model(h)


# generator2
class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()

        self.encoderG = EncoderR()
        self.encoderS = EncoderS()
        self.decoderS = DecoderS()

        hid_dim = 112
        self.tokeys = nn.Linear(hid_dim, hid_dim, bias=False)
        self.toqueries = nn.Linear(hid_dim, hid_dim, bias=False)
        self.tovalues = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, reference, sketch):
        reference_features = self.encoderG(reference)
        sketch_features = self.encoderS(sketch)

        # do self attention
        b, c, h, w = reference_features.size()
        reference_features_ = reference_features.view(b, c, h * w).contiguous().permute(0, 2, 1)
        sketch_features_ = sketch_features.view(b, c, h * w).contiguous().permute(0, 2, 1)

        sketch_features, keys, queries = self.attention(reference_features_, sketch_features_)

        #
        sketch_features = sketch_features.permute(0, 2, 1).contiguous().view(b, c, h, w)

        keys = keys.permute(0, 2, 1).contiguous().view(b, c, h, w)
        queries = queries.permute(0, 2, 1).contiguous().view(b, c, h, w)

        #
        output = self.decoderS(sketch_features)

        return output, queries , keys

    def attention(self, reference_features_, sketch_features_):
        d = reference_features_.size()[-1]

        keys = self.tokeys(reference_features_)
        queries = self.toqueries(sketch_features_)
        values = self.tovalues(reference_features_)

        queries = queries / (d ** (1 / 4))
        keys = keys / (d ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values)

        return out + sketch_features_, keys, queries

# generator
class Generator(nn.Module):
    def __init__(self, ch_style, ch_content):
        super(Generator, self).__init__()
        ch_output = 3
        base_dim = 64
        self.style_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_style, base_dim * 1, 3, 1, 1)),
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

        self.content_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(ch_content, base_dim * 1, 3, 1, 1)),
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

        self.decoder = nn.Sequential(
            spectral_norm(nn.Conv2d(base_dim * 8, base_dim * 4, 3, 1, 1)),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 4),
            ResidulBlock(base_dim * 4, base_dim * 2, sample='up'),
            ResidulBlock(base_dim * 2, base_dim * 1, sample='up'),
            ResidulBlock(base_dim * 1, base_dim * 1, sample='up'),
            nn.BatchNorm2d(base_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_dim * 1, ch_output, 3, 1, 1)),
            nn.Tanh(),
        )

    def forward(self, style, content):
        style_h = self.style_encoder(style)
        content_h = self.content_encoder(content)
        h = torch.cat([style_h, content_h], dim=1)
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
