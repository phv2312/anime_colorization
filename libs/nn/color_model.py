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
        self.hook_features = []
        self.hook_convs = []

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
            spectral_norm(nn.Conv2d(base_dim * 4, base_dim * 4, 3, 1, 1)),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

        # add hook
        for idx in [3, 7, 11]:
            self.model[idx].register_forward_hook(self.hook_f)

        for idx in [0, 4, 8, 12]:
            self.model[idx].register_forward_hook(self.hook_c)

    def hook_f(self, model, input, output):
        self.hook_features += [output]

    def hook_c(self, model, input, output):
        self.hook_convs += [output]

    def forward(self, image):
        self.hook_features = []
        self.hook_convs = []

        last_feature = self.model(image)
        b, c, h, w = last_feature.size()

        # aggregate
        previous_features = [F.interpolate(feature, size=(h, w)) for feature in self.hook_features]
        all_features = previous_features + [last_feature]
        output = torch.cat(all_features, dim=1)

        return output, self.hook_convs

class EncoderR(nn.Module):
    def __init__(self):
        super(EncoderR, self).__init__()
        base_dim = 16
        ch_style = 3
        self.hook_features = []
        self.hook_convs = []

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
            spectral_norm(nn.Conv2d(base_dim * 4, base_dim * 4, 3, 1, 1)),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

        # add hook
        for idx in [3, 7, 11]:
            self.model[idx].register_forward_hook(self.hook_f)

        for idx in [0, 4, 8, 12]:
            self.model[idx].register_forward_hook(self.hook_c)

    def hook_f(self, model, input, output):
        self.hook_features += [output]

    def hook_c(self, model, input, output):
        self.hook_convs += [output]

    def forward(self, image):
        self.hook_features = []
        self.hook_convs = []

        last_feature = self.model(image)
        b, c, h, w = last_feature.size()

        # aggregate
        previous_features = [F.interpolate(feature, size=(h, w)) for feature in self.hook_features]
        all_features = previous_features + [last_feature]
        output = torch.cat(all_features, dim=1)

        return output, self.hook_convs

# generator2
class ColorModel(nn.Module):
    def __init__(self):
        super(ColorModel, self).__init__()

        self.encoderG = EncoderR()
        self.encoderS = EncoderS()
        self.decoderS = DecoderS()

        hid_dim = 176
        self.tokeys = nn.Linear(hid_dim, hid_dim, bias=False)
        self.toqueries = nn.Linear(hid_dim, hid_dim, bias=False)
        self.tovalues = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, reference, sketch):
        ref_features, ref_all_convs = self.encoderG(reference)
        ske_features, ske_all_convs = self.encoderS(sketch)

        # do self attention
        b, c, h, w = ref_features.size()
        reference_features_ = ref_features.view(b, c, h * w).contiguous().permute(0, 2, 1)
        sketch_features_ = ske_features.view(b, c, h * w).contiguous().permute(0, 2, 1)

        ske_features, keys, queries = self.attention(reference_features_, sketch_features_)

        #
        ske_features = ske_features.permute(0, 2, 1).contiguous().view(b, c, h, w)

        keys = keys.permute(0, 2, 1).contiguous().view(b, c, h, w)
        queries = queries.permute(0, 2, 1).contiguous().view(b, c, h, w)

        #
        output = self.decoderS(ske_features, ske_all_convs)

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


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )

# discriminator
class DecoderS(nn.Module):
    def __init__(self):
        super(DecoderS, self).__init__()
        base_dim = 16

        self.resblock_h = ResidulBlock(176, base_dim * 4)
        self.dconv_up4  = up_conv(in_channels=base_dim * 8, out_channels=base_dim * 4)
        self.dconv_up3  = up_conv(in_channels=base_dim * 8, out_channels=base_dim * 4)
        self.dconv_up2  = up_conv(in_channels=base_dim * 6, out_channels=base_dim * 2)
        self.dconv_up1  = up_conv(in_channels=base_dim * 3, out_channels=base_dim)
        self.dconv_last = nn.Sequential(
            spectral_norm(nn.Conv2d(base_dim, 3, 3, 1, 1)),
            nn.Tanh()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, h, sketch_all_convs):
        conv1, conv2, conv3, conv4 = sketch_all_convs
        # b,      2b,    4b,    4b

        x = self.resblock_h(h)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim = 1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.dconv_last(x)

        return out

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

if __name__ == '__main__':
    from torchsummary import summary

    model = ColorModel()

    sketch = torch.randn((1,1,256,256))
    ref_color = torch.rand((1,3,256,256))

    output, _, _ = model(ref_color, sketch)
    print (output.size())