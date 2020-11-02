import os
import numpy as np
import cv2
import shutil
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage import measure

from component.component_wrapper import ComponentWrapper, resize_mask

def imgshow(image):
    plt.imshow(image)
    plt.show()

def random_add_border(lst_image):
    r = [int(np.random.uniform(0, 50)) for _ in range(4)]
    lst_image = [cv2.copyMakeBorder(image, r[0], r[1], r[2], r[3], borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)) for image in lst_image]

    return lst_image, r

def resize_image(image, resized_w, resized_h):
    return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)

def extract_boundary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours, find rotated rectangle, obtain four verticies, and draw
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    x, y, w, h = cv2.boundingRect(cnts[0])

    # assert
    debug_image = deepcopy(image)
    debug_image[y:y+h, x:x+w, :] = [255,255,255]
    if np.sum(255 - debug_image) == 0:
        return (x, y, x + w, y + h)
    else:
        return (0, 0, image.shape[1], image.shape[0])

def ensure_np_image(image):
    try:
        if type(image) != np.ndarray:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print('Exception: %s' % (str(e)))

    return image


def _adapt_coord(org_coords, org_shape, new_shape):
    org_h, org_w = org_shape
    new_h, new_w = new_shape

    ratio = np.array([new_h / float(org_h), new_w / float(org_w)]).reshape(-1, 2)
    new_coords = np.floor(org_coords * ratio).astype(np.int)

    return new_coords


def resize_component_and_mask(components, mask, resized_w, resized_h):
    org_shape = (mask.shape[0], mask.shape[1])
    new_shape = (resized_h, resized_w)

    new_mask = resize_mask(mask, components, size=(resized_w, resized_h))
    new_components = deepcopy(components)

    regions = list(measure.regionprops(new_mask))
    lbl2regions = {region['label']: region for region in regions}
    for component_id, component in enumerate(new_components):
        lbl = component['label']
        if lbl in lbl2regions:
            for k, v in component.items():
                new_v = lbl2regions[lbl][k] if k in lbl2regions[lbl] else v

                if k == 'image':
                    new_v = new_v.astype(np.uint8) * 255

                new_components[component_id][k] = new_v

        else:
            print ('ahihi')
            new_components[component_id]['coords'] = _adapt_coord(component['coords'], org_shape, new_shape)

    return new_components, new_mask


def mask_to_component(mask):
    components = []

    regions = list(measure.regionprops(mask))
    for region in regions:
        component_info = {
            "centroid": np.array(region.centroid),
            "area": region.area,
            "image": region.image.astype(np.uint8) * 255,
            "label": region['label'],
            "coords": region.coords,
            "bbox": region.bbox,
        }

        components.append(component_info)

    return components

def save_components(components, mask, output_folder):
    h, w = mask.shape[:2]
    if os.path.exists(output_folder):
        print ('deleting folder: %s ...' % output_folder)
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    mask_output_path = os.path.join(output_folder, 'mask.png')
    cv2.imwrite(mask_output_path, mask)

    for c_id, c in enumerate(components):
        image = c['image']
        lbl = c['label']

        output_path = os.path.join(output_folder, '_id_%d_lbl_%d.png' % (c_id, lbl))
        cv2.imwrite(output_path, image)

def save_positive_to_debug(positive_pairs, components_a, components_b, mask_a, mask_b, output_folder="debug/matching"):
    # debug
    if os.path.exists(output_folder):
        print ('deleting folder: %s ...' % output_folder)
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    for index_a, index_b in positive_pairs:
        component_a = components_a[index_a]
        component_b = components_b[index_b]

        ys, xs = np.where(mask_a == component_a['label'])
        component_a['coords'] = np.stack((ys,xs), axis=1)

        print('pair: a-%d & b-%d ...' % (index_a, index_b))

        # build color image
        component_image_colored_a = np.ones(shape=(768, 512, 3), dtype=np.uint8) * 255
        component_image_colored_b = np.ones(shape=(768, 512, 3), dtype=np.uint8) * 255

        component_image_colored_a[component_a['coords'][:,0], component_a['coords'][:,1]] = \
            np.array(component_a['color']).reshape(1,3)

        component_image_colored_b[component_b['coords'][:,0], component_b['coords'][:,1]] = \
            np.array(component_b['color']).reshape(1,3)

        _a = cv2.resize(component_image_colored_a, (128, 256), interpolation=cv2.INTER_NEAREST)
        _b = cv2.resize(component_image_colored_b, (128, 256), interpolation=cv2.INTER_NEAREST)

        output_name = os.path.join(output_folder, "a_%d_b_%d_lbl_%s.png" % (index_a, index_b, str(component_a['label'])))
        debug_image = np.concatenate([_a, _b], axis=1)

        cv2.imwrite(output_name, debug_image)

    print('mask_a shape:', mask_a.shape)
    print('mask_b shape:', mask_b.shape)
    cv2.imwrite('mask.png', np.concatenate([mask_a, mask_b], axis=1))

    print ('save debug image to %s successfully ...' % output_folder)