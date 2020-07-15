import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.nn.attention_model import AttentionModule
from libs.nn.gan_model import Generator
from libs.nn.tps_model import TPS_SpatialTransformerNetwork

def _combine_multiple_fs(fs):
    assert len(fs) > 0

    # get information of the last feature map
    _f = fs[-1]
    b, c, h, w = _f.size()

    # resize -> concat
    same_size_fs = [F.interpolate(f, size=[h,w], mode='bilinear', align_corners=True) for f in fs[:-1]] + [_f]
    return torch.cat(same_size_fs, dim=1)

class ColorModel(nn.Module):
    def __init__(self, attn_in_dim=256):
        super(ColorModel, self).__init__()

        self.tps = TPS_SpatialTransformerNetwork(F=18, I_size=(256,256), I_r_size=(256,256), I_channel_num=3)

        self.unet_sketch = Generator(ch_input=3, use_decode=True)
        self.unet_refer  = Generator(ch_input=3, use_decode=False)

        self.attention = AttentionModule(attn_in_dim)

    def forward(self, s_im, ref_im):
        """
        :param s_im & ref_augment_im: of size (256, 256, 3)
        :return:
        """

        # get augment
        ref_augment_im, G = self.tps(ref_im)

        # encoding
        sketch_f = self.unet_sketch.encode(s_im)
        refer_f  = self.unet_refer.encode(ref_augment_im)

        _, c, h, w = sketch_f.size()
        sketch_f = sketch_f.reshape(shape=(-1, c, h * w)).permute(0, 2, 1)
        refer_f  = refer_f.reshape(shape=(-1, c, h * w)).permute(0, 2, 1)

        # do attention
        sketch_f = self.attention(query_input=refer_f, key_input=sketch_f)

        sketch_f = sketch_f.reshape(shape=(-1, h, w, c)).permute(0, 3, 1, 2)
        refer_f  = refer_f.reshape(shape=(-1, h, w, c)).permute(0, 3, 1, 2)

        # decoding to get output
        output = self.unet_sketch.decode(sketch_f)

        return output, sketch_f, refer_f, G
