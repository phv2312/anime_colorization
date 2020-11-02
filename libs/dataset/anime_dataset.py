import numpy as np
import random
import os
import cv2
from torch.utils import data
import PIL.Image as Image
from skimage import measure
from torchvision.transforms import transforms

from libs.utils.dataset import load_file_from_dir, read_im
from libs.nn.tps_model import TPS
from libs.nn.self_augment.gen import Generator as SelfImageGenerator

def _random_noise_per_channel(image):
    is_pil = False
    if type(image) == Image.Image:
        image = np.array(image)
        is_pil = True

    image = image + np.random.randint(-30, 30, (1, 1, 3)).astype(np.uint8)
    image = np.clip(image, 0, 255)

    if is_pil:
        image = Image.fromarray(image)

    return image

class OneImageAnimeDataset(data.Dataset):
    def __init__(self, input_dir, feat_size=(16,16), im_size=(256,256)):
        self.color_dir  = os.path.join(input_dir, 'color')
        self.sketch_dir = os.path.join(input_dir, 'sketch')

        self.color_fns  = load_file_from_dir(self.color_dir, exts=['.png', '.jpg'])
        self.sketch_fns = load_file_from_dir(self.sketch_dir, exts=['.png', '.jpg'])

        print ("Load total: %s files ..." % len(self.color_fns))

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size[0], im_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.mask_transform  = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((feat_size[0], feat_size[1])),
            transforms.ToTensor()
        ])

        self.TPS = TPS()
        self.image_generator = SelfImageGenerator(resized_w=512, resized_h=768, min_area=10, min_size=3)

    def __len__(self):
        return len(self.color_fns)

    def __get(self, idx):
        color_fn = self.color_fns[idx]
        sketch_fn = self.sketch_fns[idx]

        target_color_image  = read_im(color_fn)
        target_sketch_image = read_im(sketch_fn, pil_mode='L')

        list_tgt, list_ref, _ = self.image_generator.process(target_color_image, target_sketch_image, crop_bbox=False)
        color_tgt, mask_tgt, components_tgt, sketch_tgt = list_tgt
        color_ref, mask_ref, components_ref, sketch_ref = list_ref
        sketch_tgt = cv2.cvtColor(sketch_tgt, cv2.COLOR_GRAY2BGR)
        sketch_ref = cv2.cvtColor(sketch_ref, cv2.COLOR_GRAY2BGR)

        mask_fg_tgt = (mask_tgt != 0).astype(np.uint8) * 255
        mask_fg_ref = (mask_ref != 0).astype(np.uint8) * 255

        # transform
        color_tgt_in = self.image_transform(color_tgt)
        color_ref_in = self.image_transform(color_ref)
        sketch_tgt_in = self.image_transform(sketch_tgt)
        sketch_ref_in = self.image_transform(sketch_ref)
        mask_tgt_in = self.mask_transform(mask_fg_tgt)
        mask_ref_in = self.mask_transform(mask_fg_ref)

        mask_tgt_in = (mask_tgt_in > 0.1).float()
        mask_ref_in = (mask_ref_in > 0.1).float()

        return (sketch_ref_in, color_ref_in, mask_ref_in), (sketch_tgt_in, color_tgt_in, mask_tgt_in)

    def __getitem__(self, idx):
        return self.__get(idx)

import matplotlib.pyplot as plt
def imgshow(im):
    plt.imshow(im)
    plt.show()

def sketch_tensor2image(sketch):
    return (sketch.detach().cpu().numpy()[0][0] * 255).astype(np.uint8)

def color_tensor2image(color, mean=0.5, std=0.5):
    color = (color + 1) / 2
    color = (color.detach().cpu()[0].permute(1,2,0).numpy() * 255).astype(np.uint8)

    return color

import torch
def revert_normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std  = torch.tensor(std).view(1,1,1,3).permute(0,3,2,1)
    mean = torch.tensor(mean).view(1,1,1,3).permute(0,3,2,1)

    return tensor_img * std + mean

def tensor2image(tensor_input, revert=True):
    if revert:
        tensor_input = revert_normalize(tensor_input.cpu())

    if len(tensor_input.shape) == 3:
        return (tensor_input[0] * 255).detach().cpu().numpy().astype(np.uint)

    return (tensor_input[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    input_dir = "/home/kan/Desktop/cinnamon/anime_colorization/simple_data/hor01_sample"
    train_dataset = OneImageAnimeDataset(input_dir, feat_size=(20,20), im_size=(320,320))
    train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)

    train_iter = iter(train_loader)

    for batch_id, batch_data in enumerate(train_iter):
        print ('processing batch_id %d...' % (batch_id + 1))

        list_ref, list_tgt = batch_data
        sketch_ref, color_ref, mask_ref = list_ref
        sketch_tgt, color_tgt, mask_tgt = list_tgt

        imgshow(color_tensor2image(sketch_ref))
        imgshow(color_tensor2image(color_ref))
        imgshow(color_tensor2image(mask_ref))

        imgshow(color_tensor2image(sketch_tgt))
        imgshow(color_tensor2image(color_tgt))
        imgshow(color_tensor2image(mask_tgt))

        continue
