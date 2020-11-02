import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import PIL.Image as Image
from collections import Counter

from augmentor.tps_wrapper import TPSWrapper
from augmentor.affine_wrapper import RandomAffineWrapper
from augmentor.wrapper import AugmentWrapper
from component.component_wrapper import ComponentWrapper, get_component_color

from utils import ensure_np_image, extract_boundary, resize_image, resize_component_and_mask, mask_to_component, random_add_border
from utils import imgshow, save_components, save_positive_to_debug

class Generator():
    def __init__(self, resized_w, resized_h,
                 min_area=0, min_size=0):
        self.resized_w = resized_w
        self.resized_h = resized_h

        self.component_wrapper = ComponentWrapper(min_area, min_size)

        self.max_ang_range   = 15
        self.max_shear_range = 0
        self.max_trans_range = 100


        self.augment_wrapper = TPSWrapper()
        self.affine_wrapper = RandomAffineWrapper()
        self.augmentor = AugmentWrapper()

    def _get_positive(self, components_a, mask_a, components_b, mask_b):
        pair = []

        for b_id, component_b in enumerate(components_b):
            a_ids = [_i for _i, _c in enumerate(components_a) if _c['label'] == component_b['label']]
            if len(a_ids) < 1: continue
            a_id = a_ids[0]

            pair.append([a_id, b_id])

        return pair

    def _get_positive_a2b(self, components_a, mask_a, components_b, mask_b, mask_a2b):
        pair = []

        for b_id, component_b in enumerate(components_b):
            b_coord = component_b['coords']

            a_lbls = mask_a2b[b_coord[:,0], b_coord[:,1]]
            a_lbl  = Counter(list(a_lbls)).most_common(1)[0][0]

            a_ids = [_i for _i, _c in enumerate(components_a) if _c['label'] == a_lbl]

            if len(a_ids) < 1: continue
            a_id = a_ids[0]

            pair.append([a_id, b_id])

        return pair

    def process(self, colored_a, sketch_a, crop_bbox=False):
        """
        # return color_a, mask_a, components_a, color_b, mask_b, components_b, positive_pairs
        """

        # . ensure np.ndarray image
        colored_a = ensure_np_image(colored_a)
        sketch_a  = ensure_np_image(sketch_a)

        org_h, org_w = colored_a.shape[:2]

        # . crop bounding_box (if any)
        if crop_bbox:
            x_min, y_min, x_max, y_max = extract_boundary(colored_a)
        else:
            x_min, y_min = 0, 0
            x_max, y_max = org_w, org_h

        colored_a = colored_a[y_min:y_max, x_min:x_max]
        sketch_a  = sketch_a[y_min:y_max, x_min:x_max]

        # . random add border
        lst_image, r = random_add_border([colored_a, sketch_a])
        colored_a, sketch_a = lst_image

        # . building mask from colored_a
        mask_a, components_a = self.component_wrapper.extract_on_color_image(colored_a)
        components_a, mask_a = resize_component_and_mask(components_a, mask_a, self.resized_w, self.resized_h)
        colored_a = resize_image(colored_a, self.resized_w, self.resized_h)
        sketch_a  = resize_image(sketch_a, self.resized_w, self.resized_h)
        get_component_color(components_a, colored_a, mode=ComponentWrapper.EXTRACT_COLOR)

        # . gen transformation parameters
        self.augmentor.gen_augment_param(key_name='a2b', key_params={
            'affine': {'ang_range':self.max_ang_range, 'shear_range':self.max_shear_range,
                       'trans_range':self.max_trans_range, 'org_shape':(self.resized_h, self.resized_w)},
            'tps': {'dshape':(self.resized_h, self.resized_w)}
        })

        # . b from a
        p = np.random.uniform(0, 1.)

        colored_b = self.augmentor.augment(key_name='a2b', image=colored_a, p=p)
        sketch_b = self.augmentor.augment(key_name='a2b', image=sketch_a, p=p)

        mask_b = self.augmentor.augment(key_name='a2b', image=mask_a, p=p)
        components_b = mask_to_component(mask_b)
        get_component_color(components_b, colored_b, mode=ComponentWrapper.EXTRACT_COLOR)
        positive_pairs = self._get_positive(components_a, mask_a, components_b, mask_b)

        return (colored_a, mask_a, components_a, sketch_a), \
               (colored_b, mask_b, components_b, sketch_b), \
               positive_pairs

if __name__ == '__main__':
    image_path = "./../../full_data/hor02_094/color/A0002.tga"
    sketch_path = "./../../full_data/hor02_094/sketch_v3/A0002.png"
    colored_a = Image.open(image_path).convert('RGB')
    sketch_a  = Image.open(sketch_path).convert('RGB')

    generator = Generator(resized_w=512, resized_h=768, min_area=20, min_size=3)

    (colored_a, mask_a, components_a, sketch_a), \
    (colored_b, mask_b, components_b, sketch_b), \
    positive_pairs = \
        generator.process(colored_a, sketch_a, crop_bbox=False)

    print (positive_pairs)

    #imgshow(np.concatenate([colored_a, colored_b], axis=1))
    #imgshow(np.concatenate([mask_a, mask_b], axis=1))
    #imgshow(np.concatenate([sketch_a, sketch_b], axis=1))

    #
    save_components(components_a, mask_a, 'debug/a')
    save_components(components_b, mask_b, 'debug/b')
    save_positive_to_debug(positive_pairs, components_a, components_b, mask_a, mask_b, "debug/matching")
    #
