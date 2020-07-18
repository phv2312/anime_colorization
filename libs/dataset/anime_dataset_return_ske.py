import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as tvF


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.84, 0.9), ratio=(1.0, 1.0), interpolation=Image.BICUBIC):
        super(RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img1, img2):
        assert img1.size == img2.size
        # fix parameter
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        # return the image with the same transformation

        img1 = tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2 = tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
        return img1, img2


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img1, img2):
        assert img1.size == img2.size

        p = random.random()
        if p < 0.5:
            img1 = tvF.hflip(img1)
        if p < 0.5:
            img2 = tvF.hflip(img2)
        return img1, img2


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img1, img2):
        assert img1.size == img2.size

        p = random.random()
        if p < 0.5:
            img1 = tvF.vflip(img1)
        if p < 0.5:
            img2 = tvF.vflip(img2)
        return img1, img2


class GeekDataset(Dataset):
    """
    > root_dir
        > 001:
            > color:  _1.png, _2.png, ...
            > sketch: _1.png, _2.png, ...
        > 002:
            > color:  _1.png, _2.png, ...
            > sketch: _1.png, _2.png, ...
    """

    def prepare_db(self, root):
        sub_dirs = os.listdir(root)
        print('Len sub_dir:', len(sub_dirs))

        sub_dir_dict = []
        sub_dir_idx = []
        corlor_fns = []
        _id = 0
        for sub_dir in sub_dirs:
            _sub_dir_full = os.path.join(root, sub_dir)

            _color_dir = os.path.join(_sub_dir_full, 'color_processed')
            if not os.path.exists(_color_dir): continue
            _color_fns = os.listdir(_color_dir)

            _sketch_dir = os.path.join(_sub_dir_full, 'sketch_processed')
            if not os.path.exists(_sketch_dir): continue
            _sketch_fns = os.listdir(_sketch_dir)

            assert len(_color_fns) == len(_sketch_fns)
            _color_fns = [os.path.join(_sub_dir_full, 'color_processed', _fn) for _fn in _color_fns]
            _sketch_fns = [os.path.join(_sub_dir_full, 'sketch_processed', _fn) for _fn in _sketch_fns]

            if len(_color_fns) >= 2:
                # adding ...
                corlor_fns += _color_fns
                sub_dir_idx += [_id] * len(_color_fns)

                sub_dir_dict += [{'color': _color_fns, 'sketch': _sketch_fns}]
                _id += 1

        return sub_dir_dict, corlor_fns, sub_dir_idx

    def __init__(self, root, resolution=256, pad_ratio=8):
        self.root = root
        self.min_crop_area = ((pad_ratio + 1) / (pad_ratio + 2)) ** 2

        #
        self.color_dir = os.path.join(self.root, 'color_processed')
        self.sketch_dir = os.path.join(self.root, 'sketch_processed')

        #
        self.sub_dirs, self.color_fns, self.sub_dir_idx = self.prepare_db(self.root)

        # transforms
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.style_img_aug = transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(self.min_crop_area, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        self.paired_aug = [
            RandomResizedCrop(resolution, scale=(self.min_crop_area, 1.0), ratio=(1.0, 1.0)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ]

    def __getitem__(self, idx):
        # idx = 3747 #2102
        # print (idx)
        """
        returns s1, s2, s3, contour such that
        s1, s2 are in the same cluster
        s3, contour are paired icon and it's contour
        note that s3 can be in different cluster
        """
        dir_id = self.sub_dir_idx[idx]
        dir_info = self.sub_dirs[dir_id]

        if len(dir_info['color']) < 2:
            return self.__getitem__((idx + 1) % len(self.color_fns))

        same_color_paths = random.choices(dir_info['color'], k=2)
        s1_path, s2_path = same_color_paths

        if random.uniform(0, 1) < 0.4:
            tmp = s1_path
            s1_path = s2_path
            s2_path = tmp

        contour_path = dir_info['sketch'][dir_info['color'].index(s1_path)]

        s1 = Image.open(s1_path).convert('RGB')
        s2 = Image.open(s2_path).convert('RGB')
        contour = Image.open(contour_path).convert('RGB')


        # for _debug_im in [s1, s2, s3, contour]:
        # 	imgshow(_debug_im)

        s1 = self.style_img_aug(s1)
        s2 = self.style_img_aug(s2)

        s1 = self.norm(s1)
        s2 = self.norm(s2)
        contour = self.norm(contour)

        return contour[:1, :, :], s2, s1, {'G': None}

    def __len__(self):
        return len(self.color_fns)
