import random
import numpy as np
import cv2
import PIL.Image as Image
from .thinplate import numpy as tps

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    #print (grid.shape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC), grid

def random_cord():
    # initialize ...
    c_src = [
        [random.uniform(0, 0.15), random.uniform(0, 0.15)],
        [random.uniform(0.85, 1), random.uniform(0, 0.1)],
        [random.uniform(0.85, 1), random.uniform(0.85, 1)],
        [random.uniform(0, 0.15), random.uniform(0.85, 1)],
    ]

    c_dst = [
        [random.uniform(0, 0.15), random.uniform(0, 0.15)],
        [random.uniform(0.85, 1), random.uniform(0, 0.1)],
        [random.uniform(0.85, 1), random.uniform(0.85, 1)],
        [random.uniform(0, 0.15), random.uniform(0.85, 1)],
    ]

    #
    _src = [random.uniform(0.1, 0.4), random.uniform(0.1, 0.4)]

    k = random.uniform(1.1, 1.3)
    _dst = [_src[0] * k, _src[1] * k]

    c_src += [_src]
    c_dst += [_dst]

    #
    _src = [random.uniform(0.5, 0.8), random.uniform(0.5, 0.8)]

    k = random.uniform(0.8, 1.)
    _dst = [_src[0] * k, _src[1] * k]

    c_src += [_src]
    c_dst += [_dst]

    return np.array(c_src), np.array(c_dst)

class TPS:
    def augment(self, input_image):
        """

        :param input_image: output image will have the same size as input image
        :return:
        """
        is_pil = False
        if type(input_image) == Image.Image:
            is_pil = True
            input_image = np.array(input_image)

        c_src, c_dst = random_cord()
        warp_image, G = warp_image_cv(input_image, c_src, c_dst, dshape=(200, 200))

        if is_pil:
            warp_image = Image.fromarray(warp_image)

        return warp_image, G
