from libs.nn.utils import *

def init_net_weight(module):
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
        nn.init.normal_(module.weight, 0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

# generator2
class ColorModel(nn.Module):
    def __init__(self, do_init_weight=True):
        super(ColorModel, self).__init__()
        # encoder
        self.feature_extraction = FeatureExtraction()
        self.adapt_layer_sketch3 = adap_layer_feat(1024)
        self.adapt_layer_sketch2 = adap_layer_feat(512)
        self.adapt_layer_sketch1 = adap_layer_feat(256)

        self.adapt_layer_reference3 = adap_layer_feat(1024)
        self.adapt_layer_reference2 = adap_layer_feat(512)
        self.adapt_layer_reference1 = adap_layer_feat(256)

        # matching layer
        self.matching_layer = matching_layer()
        self.feature_H, self.feature_W, self.beta, self.kernel_sigma = 16, 16, 50, 5
        self.find_correspondence = find_correspondence(self.feature_H, self.feature_W, self.beta, self.kernel_sigma)

        # decoder
        self.decoderS = DecoderS(in_channels=(256, 512, 1024), base_dim=64)

        if do_init_weight:
            self.adapt_layer_reference1.apply(init_net_weight)
            self.adapt_layer_sketch1.apply(init_net_weight)
            self.adapt_layer_reference2.apply(init_net_weight)
            self.adapt_layer_sketch2.apply(init_net_weight)
            self.adapt_layer_reference3.apply(init_net_weight)
            self.adapt_layer_sketch3.apply(init_net_weight)
            self.decoderS.apply(init_net_weight)

    def __sketch_extract_feature(self, input):
        feat1, feat2, feat3, _ = self.feature_extraction(input)
        feat1 = self.adapt_layer_sketch1(feat1)
        feat2 = self.adapt_layer_sketch2(feat2)
        feat3 = self.adapt_layer_sketch3(feat3)

        return feat3, [feat1, feat2, feat3]

    def __color_extract_feature(self, input):
        feat1, feat2, feat3, _ = self.feature_extraction(input)
        feat1 = self.adapt_layer_reference1(feat1)
        feat2 = self.adapt_layer_reference2(feat2)
        feat3 = self.adapt_layer_reference3(feat3)

        return feat3, [feat1, feat2, feat3]

    def __matching(self, reference_sketch_feats, target_sketch_feats):
        ref_feat1, ref_feat2, ref_feat3 = reference_sketch_feats
        tgt_feat1, tgt_feat2, tgt_feat3 = target_sketch_feats
        b, c, h, w = ref_feat3.size()

        # interpolate
        ref_feat1, ref_feat2 = [F.interpolate(ref_feat, size=(h,w), mode='bilinear', align_corners=True)
                                for ref_feat in [ref_feat1, ref_feat2]]
        tgt_feat1, tgt_feat2 = [F.interpolate(tgt_feat, size=(h,w), mode='bilinear', align_corners=True)
                                for tgt_feat in [tgt_feat1, tgt_feat2]]

        # S2T ~ ref2sketch
        corr_feat1_s2t = self.matching_layer(ref_feat1, tgt_feat1) # channel: target, spatial: source
        corr_feat2_s2t = self.matching_layer(ref_feat2, tgt_feat2) # channel: target, spatial: source
        corr_feat3_s2t = self.matching_layer(ref_feat3, tgt_feat3) # channel: target, spatial: source

        corr_S2T = corr_feat1_s2t * corr_feat2_s2t * corr_feat3_s2t
        corr_S2T = self.L2normalize(corr_S2T)

        # T2S ~ sketch2ref
        corr_feat1_t2s = corr_feat1_s2t.view(b, h*w, h*w).transpose(1, 2).view(b,h*w,h,w)  # (b, ref_h * ref_w, ske_h, ske_w)
        corr_feat2_t2s = corr_feat2_s2t.view(b, h*w, h*w).transpose(1, 2).view(b,h*w,h,w)  # (b, ref_h * ref_w, ske_h, ske_w)
        corr_feat3_t2s = corr_feat3_s2t.view(b, h*w, h*w).transpose(1, 2).view(b,h*w,h,w)  # (b, ref_h * ref_w, ske_h, ske_w)

        corr_T2S = corr_feat1_t2s * corr_feat2_t2s * corr_feat3_t2s
        corr_T2S = self.L2normalize(corr_T2S)

        return corr_S2T, corr_T2S

    def __do_attention(self, corr_S2T, reference_color_feat, target_sketch_feat):
        b, c, h, w = corr_S2T.size()

        corr_S2T_ = corr_S2T
        attention = self.find_correspondence.softmax_with_temperature(corr_S2T_, beta=self.beta, d=1)

        context_vector = torch.bmm(
            attention.view(b, h*w, h*w),
            reference_color_feat.view(b, -1, h * w).contiguous().transpose(1, 2)
        )

        target_sketch_feat_combine = context_vector + target_sketch_feat.view(b, -1, h * w).contiguous().transpose(1, 2)
        target_sketch_feat_combine = target_sketch_feat_combine.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return target_sketch_feat_combine

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def __encode(self, reference_color, reference_sketch, target_sketch):
        reference_color_feat, reference_color_feats = self.__color_extract_feature(reference_color)
        reference_sketch_feat, reference_sketch_feats = self.__sketch_extract_feature(reference_sketch)
        target_sketch_feat, target_sketch_feats = self.__sketch_extract_feature(target_sketch)

        corr_S2T, corr_T2S = self.__matching(reference_sketch_feats, target_sketch_feats)
        target_sketch_feat_combine = self.__do_attention(corr_S2T, reference_color_feat, target_sketch_feat)

        return target_sketch_feat_combine, \
               (corr_S2T, corr_T2S), \
               (reference_color_feats, reference_sketch_feats, target_sketch_feats)

    def __decode(self, target_sketch_feat, target_sketch_feats):
        output = self.decoderS(target_sketch_feat, target_sketch_feats)
        return output

    def forward(self, reference_color, reference_sketch, target_sketch, mode='train', GT_src_mask=None, GT_tgt_mask=None):
        # encoding
        target_sketch_feat_combine, corrs, feats = self.__encode(reference_color, reference_sketch, target_sketch)
        corr_S2T, corr_T2S = corrs
        reference_color_feats, reference_sketch_feats, target_sketch_feats = feats

        # decoding (need debug again)
        output = self.__decode(target_sketch_feat_combine, target_sketch_feats)

        semantic_output = {}
        if mode=='train':
            # establish correspondences
            grid_S2T, flow_S2T, smoothness_S2T = self.find_correspondence(corr_S2T, GT_src_mask)
            grid_T2S, flow_T2S, smoothness_T2S = self.find_correspondence(corr_T2S, GT_tgt_mask)

            # estimate warped masks
            warped_src_mask = F.grid_sample(GT_tgt_mask, grid_S2T, mode='bilinear')
            warped_tgt_mask = F.grid_sample(GT_src_mask, grid_T2S, mode='bilinear')

            # estimate warped flows
            warped_flow_S2T = -F.grid_sample(flow_T2S, grid_S2T, mode='bilinear') * GT_src_mask
            warped_flow_T2S = -F.grid_sample(flow_S2T, grid_T2S, mode='bilinear') * GT_tgt_mask
            flow_S2T = flow_S2T * GT_src_mask
            flow_T2S = flow_T2S * GT_tgt_mask

            semantic_output = {
                'est_src_mask': warped_src_mask, 'smoothness_S2T': smoothness_S2T, 'grid_S2T': grid_S2T,
                'est_tgt_mask': warped_tgt_mask, 'smoothness_T2S': smoothness_T2S, 'grid_T2S': grid_T2S,
                'flow_S2T': flow_S2T, 'flow_T2S': flow_T2S,
                'warped_flow_S2T': warped_flow_S2T, 'warped_flow_T2S': warped_flow_T2S
            }

        return output, semantic_output

# discriminator
class DecoderS(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024), base_dim=64):
        super(DecoderS, self).__init__()

        self.resblock_h = ResidulBlock(in_channels[-1], base_dim * 4)
        self.dconv_up3  = up_conv(in_channels=base_dim * 4 + in_channels[-2], out_channels=base_dim * 4)
        self.dconv_up2  = up_conv(in_channels=base_dim * 4 + in_channels[-3], out_channels=base_dim * 3)
        self.dconv_up1  = up_conv(in_channels=base_dim * 3, out_channels=base_dim)
        self.dconv_last = nn.Sequential(
            spectral_norm(nn.Conv2d(base_dim, 3, 3, 1, 1)),
            nn.Tanh()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, h, target_sketch_features):
        feat1, feat2, _ = target_sketch_features

        x = self.resblock_h(h)
        x = self.upsample(x) #.125
        x = torch.cat([x, feat2], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x) #.25
        x = torch.cat([x, feat1], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x) #.5

        x = self.dconv_up1(x)
        x = self.upsample(x) #1

        x = self.dconv_last(x)

        return x


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
    model = ColorModel()

    reference_color  = torch.rand((1,3,320,320))
    reference_sketch = torch.rand((1,3,320,320))
    target_sketch    = torch.rand((1,3,320,320))

    output, _ = model(reference_color, reference_sketch, target_sketch, mode='!train')
    print ('output size:', output.size())