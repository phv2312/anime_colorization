import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class PretrainedModel(nn.Module):
    def __init__(self, normalize, content_layer, style_layer):
        super(PretrainedModel, self).__init__()
        self.cnn = models.vgg19(pretrained=True).features.eval()

        self.content_layer = content_layer
        self.style_layer   = style_layer
        self.normalize = normalize

    def forward(self, input):
        cnn = copy.deepcopy(self.cnn)
        content_losses = []
        style_losses   = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalize)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layer:
                # add content loss
                f = model(input).detach()
                content_losses.append(f)

            if name in self.style_layer:
                # add style loss
                f = model(input).detach()
                style_losses.append(f)

        return content_losses, style_losses



class L1StyleContentLoss(nn.Module):
    def __init__(self, content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        super(L1StyleContentLoss, self).__init__()

        # desired depth layers to compute style/content losses :
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.pretrained_model = PretrainedModel(Normalization(), content_layers, style_layers)

    def forward(self, predict, target):
        """

        :param predict: of size (b, c, h, w)
        :param target: of size (b, c, h, w)
        :return: style, perceptual, l1
        """

        predict.data.clamp_(0, 1)
        target.data.clamp_(0, 1)
        with torch.no_grad():
            #
            pred_content_fs, pred_style_fs = self.pretrained_model(predict)
            tagt_content_fs, tagt_style_fs = self.pretrained_model(target)

            #
            content_loss = 0.
            for pred_content_f, tagt_content_f in zip(pred_content_fs, tagt_content_fs):
                content_loss += F.mse_loss(pred_content_f, tagt_content_f)

            style_loss = 0.
            for pred_style_f, tagt_style_f in zip(pred_style_fs, tagt_style_fs):
                style_loss += F.mse_loss(gram_matrix(pred_style_f), gram_matrix(tagt_style_f))

            l1_score = F.l1_loss(predict, target)

            return style_loss, content_loss, l1_score