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

class FeatureExtraction2(nn.Module):
    def __init__(self):
        super(FeatureExtraction2, self).__init__()

        model = models.vgg19(pretrained=True)
        self.layer = model.features

        self.extracted_layer_name = ['1', '6', '11', '20', '29']

    def forward(self, x):
        outputs = []
        for _id, module in self.layer._modules.items():
            x = module(x)
            if str(_id) in self.extracted_layer_name:
                outputs.append(x)

        return outputs


class FeatureExtraction(nn.Module):
    """
        Feature extraction model for Perceptual Loss and Style Loss
        Use VGG19 to extract features.
    """

    def __init__(self, model):
        super(FeatureExtraction, self).__init__()
        # For VGG19 network
        self.model = nn.Sequential(*list(*list(model.children())[:1])[:32])
        self.feature_blobs = []
        self.register_hook()

        # # freeze weights
        # for layer in self.model.parameters():
        #     layer.requires_grad = False

    def hook_feature(self, module, input, output):
        self.feature_blobs.append(output.data)

    def register_hook(self):
        self.model._modules.get("1").register_forward_hook(self.hook_feature)
        self.model._modules.get("6").register_forward_hook(self.hook_feature)
        self.model._modules.get("11").register_forward_hook(self.hook_feature)
        self.model._modules.get("20").register_forward_hook(self.hook_feature)
        self.model._modules.get("29").register_forward_hook(self.hook_feature)

    def forward(self, x):
        self.feature_blobs = []
        self.model(x)
        return self.feature_blobs


class L1StyleContentLoss(nn.Module):
    def __init__(self, content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        super(L1StyleContentLoss, self).__init__()

        # desired depth layers to compute style/content losses :
        self.content_layers = content_layers
        self.style_layers = style_layers

        self.feature_extractor = FeatureExtraction2() #FeatureExtraction(models.vgg19(pretrained=True).eval())

    def forward(self, predict, target):
        """

        :param predict: of size (b, c, h, w)
        :param target: of size (b, c, h, w)
        :return: style, perceptual, l1
        """
        predict_features = self.feature_extractor(predict)
        target_features  = self.feature_extractor(target)

        content_loss = 0.
        style_loss = 0.
        for predict_feature, target_feature in zip(predict_features, target_features):
            content_loss += F.l1_loss(predict_feature, target_feature)
            style_loss += F.l1_loss(gram_matrix(predict_feature), gram_matrix(target_feature))

        l1_loss = F.l1_loss(predict, target)

        return style_loss, content_loss, l1_loss

if __name__ == '__main__':
    feature_extract = FeatureExtraction2().eval()
    feature_extract(torch.rand(size=(1,3,256,256)))
