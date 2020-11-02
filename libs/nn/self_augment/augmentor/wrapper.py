import numpy as np
from copy import deepcopy

from .affine_wrapper import RandomAffineWrapper
from .tps_wrapper import TPSWrapper


class AugmentWrapper:
    def __init__(self):
        self.random_affine = RandomAffineWrapper()
        self.tps = TPSWrapper()

        self.alls = {
            'tps': self.tps,
            'affine': self.random_affine
        }

    def gen_augment_param(self, key_params, key_name):
        for aug_name, aug in self.alls.items():
            _key_param = key_params[aug_name]
            _key_param['key_name'] = key_name

            aug.gen(**_key_param)

    def augment(self, key_name, image, p):
        output = deepcopy(image)

        all = {}
        if p < 0.65: all = {'tps': self.tps}
        else: all = {'affine': self.random_affine}

        for aug_name, aug in all.items():
            output = aug.augment(output, key_name)

        return output