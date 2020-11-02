import os
import glob
import numpy as np
import PIL.Image as Image

def read_im(im_path, pil_mode='RGB'):
    return np.array(Image.open(im_path).convert(pil_mode))

def load_file_from_dir(dir, exts=['.jpg', '.png']):
    fns = []

    for ext in exts:
        _fns = glob.glob(os.path.join(dir, "*%s" % ext))
        fns += list(_fns)

    return fns