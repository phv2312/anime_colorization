import os
import glob
import numpy as np
import cv2
import skimage.measure as measure
import skimage.feature
from math import copysign, log10
from PIL import Image
from natsort import natsorted

def get_moment_features(components, mask):
    features = np.zeros([mask.shape[0], mask.shape[1], 8])

    for component in components:
        image = component["image"]
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        moments = cv2.moments(image)
        moments = cv2.HuMoments(moments)[:, 0]
        for i in range(0, 7):
            if moments[i] == 0:
                continue
            moments[i] = -1 * copysign(1.0, moments[i]) * log10(abs(moments[i]))

        moments = np.append(moments, component["area"] / 200000.0)
        coords = np.nonzero(mask == component["label"])
        features[coords[0], coords[1], :] = moments

    features = np.transpose(features, (2, 0, 1))
    return features


def build_neighbor_graph(mask):
    max_level = mask.max() + 1
    matrix = skimage.feature.greycomatrix(mask, [1, 3], [0], levels=max_level)
    matrix = np.sum(matrix, axis=(2, 3))
    graph = np.zeros((max_level, max_level))

    for i in range(1, max_level):
        for j in range(1, max_level):
            if matrix[i, j] > 0:
                graph[i, j] = 1
                graph[j, i] = 1
    return graph


class ComponentWrapper:
    EXTRACT_COLOR = "extract_color"
    EXTRACT_SKETCH = "extract_sketch"

    def __init__(self, min_area=10, min_size=1):
        self.min_area = min_area
        self.min_size = min_size
        self.bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 15, 255]]

    def extract_on_color_image(self, input_image):
        b, g, r = cv2.split(input_image)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)

        index = 0
        components = {}
        mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)

        # Pre-processing image
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        # Get number of colors in image
        uniques = np.unique(processed_image)

        for unique in uniques:
            # Ignore sketch (ID of background is 255)
            if unique in self.bad_values:
                continue

            rows, cols = np.where(processed_image == unique)
            # Mask
            image_temp = np.zeros_like(processed_image)
            image_temp[rows, cols] = 255
            image_temp = np.array(image_temp, dtype=np.uint8)

            # Connected components
            labels = measure.label(image_temp, connectivity=1, background=0)
            regions = measure.regionprops(labels, intensity_image=processed_image)

            for region in regions:
                if region.area < self.min_area:
                    continue
                if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                    continue
                if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                    continue

                if unique == 23117055 and [0, 0] in region.coords:
                    continue

                components[index] = {
                    "centroid": np.array(region.centroid),
                    "area": region.area,
                    "image": region.image.astype(np.uint8) * 255,
                    "label": index + 1,
                    "coords": region.coords,
                    "bbox": region.bbox,
                    "min_intensity": region.min_intensity,
                    "mean_intensity": region.mean_intensity,
                    "max_intensity": region.max_intensity,
                }
                mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
                index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def extract_on_sketch_v3(self, sketch):
        binary = cv2.threshold(sketch, 100, 255, cv2.THRESH_BINARY)[1]
        labels = measure.label(binary, connectivity=1, background=0)
        regions = measure.regionprops(labels, intensity_image=sketch)

        index = 0
        mask = np.zeros((sketch.shape[0], sketch.shape[1]), dtype=np.int)
        components = dict()

        for region in regions[1:]:
            if region.area < self.min_area:
                continue
            if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                continue
            if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                continue

            components[index] = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": region.image.astype(np.uint8) * 255,
                "label": index + 1,
                "coords": region.coords,
                "bbox": region.bbox,
                "min_intensity": region.min_intensity,
                "mean_intensity": region.mean_intensity,
                "max_intensity": region.max_intensity,
            }
            mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
            index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def process(self, input_image, sketch, method):
        assert len(cv2.split(input_image)) == 3, "Input image must be RGB, got binary"
        assert method in [self.EXTRACT_COLOR, self.EXTRACT_SKETCH]

        if method == self.EXTRACT_COLOR:
            mask, components = self.extract_on_color_image(input_image)
        else:
            mask, components = self.extract_on_sketch_v3(sketch)
        return mask, components


def get_component_color(components, color_image, mode=ComponentWrapper.EXTRACT_SKETCH):
    if mode == ComponentWrapper.EXTRACT_COLOR:
        for component in components:
            index = len(component["coords"]) // 2
            coord = component["coords"][index]
            color = color_image[coord[0], coord[1]].tolist()
            component["color"] = color

    elif mode == ComponentWrapper.EXTRACT_SKETCH:
        for component in components:
            coords = component["coords"]
            points = color_image[coords[:, 0], coords[:, 1]]

            unique, counts = np.unique(points, return_counts=True, axis=0)
            max_index = np.argmax(counts)
            color = unique[max_index].tolist()
            component["color"] = color
    return


def rectify_mask(mask, component, ratio):
    coords = component["coords"]
    new_coords = np.array([[int(coord[0] * ratio[0]), int(coord[1] * ratio[1])] for coord in coords])
    new_coords = list(np.unique(new_coords, axis=0).tolist())

    count = 0
    mid_index = int(len(new_coords) / 2)
    new_area = {component["label"]: len(new_coords)}

    for i in range(0, mid_index + 1):
        offsets = [1] if i == 0 else [-1, 1]
        for j in offsets:
            index = mid_index + i * j
            if index >= len(new_coords):
                continue
            coord = new_coords[index]

            if mask[coord[0], coord[1]] == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
                continue

            label = mask[coord[0], coord[1]]
            if label not in new_area:
                new_area[label] = np.count_nonzero(mask == label)

            if new_area[label] > new_area[component["label"]] * 5:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
            elif new_area[label] > 1 and count == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
    return mask


def resize_mask(mask, components, size):
    new_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    ratio = (size[1] / mask.shape[0], size[0] / mask.shape[1])

    old_labels = np.unique(mask).tolist()
    new_labels = np.unique(new_mask).tolist()
    removed_labels = [i for i in old_labels if (i not in new_labels) and (i > 0)]

    for i in removed_labels:
        component = components[i - 1]
        new_mask = rectify_mask(new_mask, component, ratio)

    assert len(np.unique(mask)) == len(np.unique(new_mask)), 'len old mask: %d vs len new mask: %d' % (len(np.unique(mask)), len(np.unique(new_mask)))
    return new_mask


def main():
    root_dir = "D:/Data/GeekToys/coloring_data/simple_data"
    output_dir = "D:/Data/GeekToys/output/rules"

    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    component_wrapper = ComponentWrapper(min_area=15, min_size=3)

    bug_names = natsorted(glob.glob(os.path.join("D:/Data/GeekToys/output/bug", "*.png")))
    bug_names = [os.path.splitext(os.path.basename(path))[0] for path in bug_names]

    for character_dir in character_dirs:
        character_name = os.path.basename(character_dir)
        paths = natsorted(glob.glob(os.path.join(character_dir, "color", "*.tga")))

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            full_name = "%s_%s" % (character_name, name)
            print(full_name)

            if "hor01_047_k_C_C0002" not in full_name:
                continue

            color_image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            output_mask, output_components = component_wrapper.process(color_image)
            get_component_color(output_components, color_image)

            new_mask = resize_mask(output_mask, output_components, (768, 512))
            white_mask = np.where(output_mask == 0, np.zeros_like(output_mask), np.full_like(output_mask, 255))
            graph = build_neighbor_graph(new_mask)

            if len(np.unique(output_mask)) == len(np.unique(new_mask)):
                print(len(np.unique(output_mask)), len(np.unique(new_mask)), output_mask.shape, new_mask.shape)
                cv2.imwrite(os.path.join(output_dir, "%s.png" % full_name), output_mask)
                cv2.imwrite(os.path.join(output_dir, "%s_.png" % full_name), white_mask)
    return


if __name__ == "__main__":
    main()
