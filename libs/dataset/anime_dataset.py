from torch.utils import data

from libs.utils.dataset import load_file_from_dir, read_im

class OneImageAnimeDataset(data.Dataset):
    def __init__(self, input_dir, transform=None, is_train=True):
        self.input_dir = input_dir
        self.input_fns = load_file_from_dir(input_dir, exts=['.png', '.jpg'])
        print ("Load total: %s files ..." % len(self.input_fns))

        self.transform = transform if transform is not None else lambda k:k
        self.is_train = is_train

    def __len__(self):
        return len(self.input_fns)

    def __getitem__(self, idx):
        # input_fn  = self.input_fns[idx]
        # image = read_im(input_fn)
        #
        # tensor_image = self.transform(image)
        # return tensor_image

        ref_image = read_im(im_path="/home/kan/Desktop/Cinnamon/gan/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions/geek/full_data/hor02/hor02_31/color_processed/A0001.png")
        ske_iamge = read_im(im_path="/home/kan/Desktop/Cinnamon/gan/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions/geek/full_data/hor02/hor02_31/sketch_processed/A0001.png")

        return self.transform(ske_iamge), self.transform(ref_image)

