import torch.utils.data as data
import torchvision.transforms as transforms
from natsort import natsorted
import glob
import os
from PIL import Image, ImageCms
import random

def get_image_by_index(items, index):
    if index is None:
        return None
    path = items[index]
    image = Image.open(path).convert("RGB")
    return image


def do_transform(load_size, method=Image.BICUBIC, same_image=False, grayscale=False, convert=True, resize=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if resize:
        transform_list.append(transforms.Resize(load_size, method))
    if same_image:
        transform_list.append(transforms.RandomAffine(degrees=24, translate=(0.1, 0.1), fillcolor=(255, 255, 255)))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class AnimeDataset(data.Dataset):
    def __init__(self, root_dir, load_size=(256,256), num_rand=69):
        super(AnimeDataset, self).__init__()
        random.seed(num_rand)
        self.root_dir = root_dir
        self.load_size = load_size

        self.paths = {}
        self.lengths = {}
        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))

        for sub_dir in dirs:
            dir_name = os.path.basename(sub_dir)
            self.paths[dir_name] = {}

            for set_name in ["sketch_v3", "color"]:
                paths = []
                for sub_type in ["png", "jpg", "tga"]:
                    paths.extend(glob.glob(os.path.join(sub_dir, set_name, "*.%s" % sub_type)))
                self.paths[dir_name][set_name] = natsorted(paths)

            self.lengths[dir_name] = len(self.paths[dir_name]["color"])

        return

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def __getitem__(self, index):
        name = None

        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        # pick sketch's path for visualization purpose
        sketch_path = self.paths[name]["sketch_v3"][index]

        # pick ref index
        ref_index = random.randint(0, len(self.paths[name]["color"]) - 1)

        # get path of ref, sketch, groundtruth
        ref = get_image_by_index(self.paths[name]["color"], ref_index)
        sketch = get_image_by_index(self.paths[name]["sketch_v3"], index)
        gt = get_image_by_index(self.paths[name]["color"], index)

        color_tf = do_transform(self.load_size)
        sketch_tf = do_transform(self.load_size, grayscale=True)
        same_image_tf = do_transform(self.load_size, same_image=True)

        if ref_index == index:
            ref = same_image_tf(ref)
        else:
            ref = color_tf(ref)
        sketch = sketch_tf(sketch)
        gt = color_tf(gt)

        return sketch, gt, ref, {'sketch_path': sketch_path, 'G': []}