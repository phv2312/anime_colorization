import numpy as np
import random
import os
from torch.utils import data
import PIL.Image as Image

from libs.utils.dataset import load_file_from_dir, read_im
from libs.nn.tps_model import TPS

def _random_noise_per_channel(image):
    is_pil = False
    if type(image) == Image.Image:
        image = np.array(image)
        is_pil = True

    image = image + np.random.randint(-50, 50, (1, 1, 3)).astype(np.uint8)
    image = np.clip(image, 0, 255)

    if is_pil:
        image = Image.fromarray(image)

    return image

class OneImageAnimeDataset(data.Dataset):
    def __init__(self, input_dir, transform=None, is_train=True):
        self.color_dir  = os.path.join(input_dir, 'color')
        self.sketch_dir = os.path.join(input_dir, 'sketch')

        self.color_fns  = load_file_from_dir(self.color_dir, exts=['.png', '.jpg'])
        self.sketch_fns = load_file_from_dir(self.sketch_dir, exts=['.png', '.jpg'])

        print ("Load total: %s files ..." % len(self.color_fns))

        self.transform = transform if transform is not None else lambda k:k
        self.is_train = is_train

        self.TPS = TPS()

    def __len__(self):
        return len(self.color_fns)

    def __getitem__(self, idx):
        color_fn  = self.color_fns[idx]
        sketch_fn = self.sketch_fns[idx]

        color_image = read_im(im_path=color_fn)
        #if random.uniform(0, 1.) < 0.3: color_image = _random_noise_per_channel(color_image)

        augment_color_image, G = self.TPS.augment(color_image)
        sketch_image = read_im(im_path=sketch_fn)

        if self.transform is not None:
            augment_color_image = self.transform(augment_color_image)
            sketch_image = self.transform(sketch_image)
            color_image = self.transform(color_image)

        return sketch_image[:1,:,:], color_image, augment_color_image, {'G': G}
