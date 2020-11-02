import cv2
import numpy as np

class RandomAffineWrapper():
    def __init__(self):
        self.matrix_dct = {}

    def gen(self, ang_range, shear_range, trans_range, org_shape, key_name):
        rot_matrix, trans_matrix, shear_matrix = self.affine_transform_image(org_shape, ang_range, shear_range, trans_range)
        self.matrix_dct[key_name] = [rot_matrix, trans_matrix, shear_matrix, org_shape]

    def augment(self, img, key_name):
        border_mode = cv2.BORDER_REPLICATE
        flags = cv2.INTER_NEAREST

        rot_matrix, trans_matrix, shear_matrix, org_shape = self.matrix_dct[key_name]

        cols = org_shape[1]
        rows = org_shape[0]

        img = cv2.warpAffine(img, rot_matrix, (cols, rows), borderMode=border_mode, flags=flags)
        img = cv2.warpAffine(img, trans_matrix, (cols, rows), borderMode=border_mode, flags=flags)
        img = cv2.warpAffine(img, shear_matrix, (cols, rows), borderMode=border_mode, flags=flags)

        return img

    def affine_transform_image(self, org_shape, ang_range, shear_range, trans_range):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.

        A Random uniform distribution is used to generate different parameters for transformation

        '''
        # rotation
        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows, cols = org_shape[:2]
        rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)

        return rot_M, trans_M, shear_M