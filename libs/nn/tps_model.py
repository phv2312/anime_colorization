import random
import numpy as np
import cv2
import PIL.Image as Image
from math import floor
from .thinplate import numpy as tps

import matplotlib.pyplot as plt
def imgshow(im):
    plt.imshow(im)
    plt.show()

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
        warp_image, G = warp_image_cv(input_image, c_src, c_dst, dshape=(256, 256))

        if is_pil:
            warp_image = Image.fromarray(warp_image)

        return warp_image, G

    def test_invert_transform(self, input_image):
        warp_image, G = self.augment(input_image)
        imgshow(warp_image)

        # revert back to input from warp
        G = G.T

        mapx, mapy = tps.tps_grid_to_remap(G, warp_image.shape)
        ahihi = cv2.remap(warp_image, mapx, mapy, cv2.INTER_CUBIC)

        imgshow(ahihi)

    def visualize(self, input_image):
        warp_image, G = self.augment(input_image)

        h, w = warp_image.shape[:2]
        imgshow(input_image)
        imgshow(warp_image)

        for ih in range(60, h, 20):
            for iw in range(0, w, 20):
                _input_image = input_image.copy()
                _warp_image  = warp_image.copy()

                ix, iy = G[ih, iw]
                print (ix, iy)

                if ix < 0.01 or iy <= 0.01:
                    print('continue')
                    continue

                ix = np.clip(ix, 0., 1.)
                iy = np.clip(iy, 0., 1.)

                ix *= w #((ix + 1) / 2) * (w - 1)
                iy *= h

                # bot-left
                ix_nw = int(floor(ix))
                iy_nw = int(floor(iy))

                # top-right
                ix_se = ix_nw + 1
                iy_se = iy_nw + 1

                # conduct positive ids
                positive_ids = []
                for _x in range(ix_nw, ix_se + 1):
                    for _y in range(iy_nw, iy_se + 1):

                        if 0 <= _x <= w and 0 <= _y <= h:
                            positive_ids += [(_x, _y)]

                # visualize
                cv2.circle(_warp_image, (iw, ih), radius=3, color=(255,255,0), thickness=3)
                print (positive_ids)
                for _x, _y in positive_ids:
                    cv2.circle(_input_image, (_x, _y), radius=2, color=(255,0,255), thickness=3)

                imgshow(np.concatenate([_input_image, _warp_image], axis=1))

if __name__ == '__main__':
    input_image= cv2.imread("/home/kan/Desktop/Cinnamon/gan/self_augment_color/sample_dataset/1/color/A0001.png")
    input_image = cv2.resize(input_image, dsize=(256,256))
    TPS().visualize(input_image)


