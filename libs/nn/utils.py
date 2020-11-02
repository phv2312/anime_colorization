import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models

import numpy as np

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = models.resnet101(pretrained=True)
        resnet_feature_layers = ['conv1',
                                 'bn1',
                                 'relu',
                                 'maxpool',
                                 'layer1',
                                 'layer2',
                                 'layer3',
                                 'layer4', ]
        layer1 = 'layer1'
        layer2 = 'layer2'
        layer3 = 'layer3'
        layer4 = 'layer4'
        layer1_idx = resnet_feature_layers.index(layer1)
        layer2_idx = resnet_feature_layers.index(layer2)
        layer3_idx = resnet_feature_layers.index(layer3)
        layer4_idx = resnet_feature_layers.index(layer4)
        resnet_module_list = [self.model.conv1,
                              self.model.bn1,
                              self.model.relu,
                              self.model.maxpool,
                              self.model.layer1,
                              self.model.layer2,
                              self.model.layer3,
                              self.model.layer4]
        self.layer1 = nn.Sequential(*resnet_module_list[:layer1_idx + 1])
        self.layer2 = nn.Sequential(*resnet_module_list[layer1_idx + 1:layer2_idx + 1])
        self.layer3 = nn.Sequential(*resnet_module_list[layer2_idx + 1:layer3_idx + 1])
        self.layer4 = nn.Sequential(*resnet_module_list[layer3_idx + 1:layer4_idx + 1])
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False

    def forward(self, image_batch):
        layer1_feat = self.layer1(image_batch)
        layer2_feat = self.layer2(layer1_feat)
        layer3_feat = self.layer3(layer2_feat)
        layer4_feat = self.layer4(layer3_feat)
        return layer1_feat, layer2_feat, layer3_feat, layer4_feat


class matching_layer(nn.Module):
    def __init__(self):
        super(matching_layer, self).__init__()
        self.relu = nn.ReLU()

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, feature1, feature2):
        feature1 = self.L2normalize(feature1)
        feature2 = self.L2normalize(feature2)
        b, c, h1, w1 = feature1.size()
        b, c, h2, w2 = feature2.size()
        feature1 = feature1.view(b, c, h1 * w1)
        feature2 = feature2.view(b, c, h2 * w2)
        corr = torch.bmm(feature2.transpose(1, 2), feature1)
        corr = corr.view(b, h2 * w2, h1, w1)  # Channel : target // Spatial grid : source
        corr = self.relu(corr)
        return corr


class find_correspondence(nn.Module):
    def __init__(self, feature_H, feature_W, beta, kernel_sigma):
        super(find_correspondence, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.kernel_sigma = kernel_sigma

        # regular grid / [-1,1] normalized
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, feature_W),
                                               np.linspace(-1, 1, feature_H))  # grid_X & grid_Y : feature_H x feature_W
        self.grid_X = torch.tensor(self.grid_X, dtype=torch.float, requires_grad=False).to(device)
        self.grid_Y = torch.tensor(self.grid_Y, dtype=torch.float, requires_grad=False).to(device)

        # kernels for computing gradients
        self.dx_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 1, 3).expand(1, 2,
                                                                                                                  1,
                                                                                                                  3).to(
            device)
        self.dy_kernel = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=False).view(1, 1, 3, 1).expand(1, 2,
                                                                                                                  3,
                                                                                                                  1).to(
            device)

        # 1-d indices for generating Gaussian kernels
        self.x = np.linspace(0, feature_W - 1, feature_W)
        self.x = torch.tensor(self.x, dtype=torch.float, requires_grad=False).to(device)
        self.y = np.linspace(0, feature_H - 1, feature_H)
        self.y = torch.tensor(self.y, dtype=torch.float, requires_grad=False).to(device)

        # 1-d indices for kernel-soft-argmax / [-1,1] normalized
        self.x_normal = np.linspace(-1, 1, feature_W)
        self.x_normal = torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False).to(device)
        self.y_normal = np.linspace(-1, 1, feature_H)
        self.y_normal = torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False).to(device)

    def apply_gaussian_kernel(self, corr, sigma=5):
        b, hw, h, w = corr.size()

        idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
        idx_y = (idx // w).view(b, 1, 1, h, w).float()
        idx_x = (idx % w).view(b, 1, 1, h, w).float()

        x = self.x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
        y = self.y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel.view(b, hw, h, w)

        return gauss_kernel * corr

    def softmax_with_temperature(self, x, beta, d=1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(beta * x)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def kernel_soft_argmax(self, corr):
        b, _, h, w = corr.size()

        corr = self.apply_gaussian_kernel(corr, sigma=self.kernel_sigma)
        corr = self.softmax_with_temperature(corr, beta=self.beta, d=1)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def get_flow_smoothness(self, flow, GT_mask):
        flow_dx = F.conv2d(F.pad(flow, (1, 1, 0, 0)), self.dx_kernel) / 2  # (padLeft, padRight, padTop, padBottom)
        flow_dy = F.conv2d(F.pad(flow, (0, 0, 1, 1)), self.dy_kernel) / 2  # (padLeft, padRight, padTop, padBottom)

        flow_dx = torch.abs(flow_dx) * GT_mask  # consider foreground regions only
        flow_dy = torch.abs(flow_dy) * GT_mask

        smoothness = torch.cat((flow_dx, flow_dy), 1)
        return smoothness

    def forward(self, corr, GT_mask=None):
        b, _, h, w = corr.size()
        grid_X = self.grid_X.expand(b, h, w)  # x coordinates of a regular grid
        grid_X = grid_X.unsqueeze(1)  # b x 1 x h x w
        grid_Y = self.grid_Y.expand(b, h, w)  # y coordinates of a regular grid
        grid_Y = grid_Y.unsqueeze(1)

        if self.beta is not None:
            grid_x, grid_y = self.kernel_soft_argmax(corr)
        else:  # discrete argmax
            _, idx = torch.max(corr, dim=1)
            grid_x = idx % w
            grid_x = (grid_x.float() / (w - 1) - 0.5) * 2
            grid_y = idx // w
            grid_y = (grid_y.float() / (h - 1) - 0.5) * 2

        grid = torch.cat((grid_x.permute(0, 2, 3, 1), grid_y.permute(0, 2, 3, 1)),
                         3)  # 2-channels@3rd-dim, first channel for x / second channel for y
        flow = torch.cat((grid_x - grid_X, grid_y - grid_Y),
                         1)  # 2-channels@1st-dim, first channel for x / second channel for y

        if GT_mask is None:  # test
            return grid, flow
        else:  # train
            if type(GT_mask) is list:
                smoothness = [self.get_flow_smoothness(flow, _GT_mask) for _GT_mask in GT_mask]
            else:
                smoothness = self.get_flow_smoothness(flow, GT_mask)
            return grid, flow, smoothness

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

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )

class adap_layer_feat(nn.Module):
    def __init__(self, in_channel):
        super(adap_layer_feat, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=1, padding=2),
        #     nn.BatchNorm2d(in_channel),
        #     nn.ReLU()
        # )

    def forward(self, feature):
        feature = feature + self.conv1(feature)
        #feature = feature + self.conv2(feature)
        return feature

class adap_layer_feat3(nn.Module):
    def __init__(self):
        super(adap_layer_feat3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

    def forward(self, feature):
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        return feature