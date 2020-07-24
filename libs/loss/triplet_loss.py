import torch
import random
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F

def _calc_dot_product(elem1, elem2):
    return torch.matmul(elem1, elem2.t())

def _to_ys(idx):
    return [_[1] for _ in idx]

def _to_xs(idx):
    return [_[0] for _ in idx]

class GridVisualize():
    def __init__(self, org_size, feature_size):
        self.org_w, self.org_h = org_size
        self.feature_w, self.feature_h = feature_size
        self.r = self.org_w // self.feature_w

    def to_video(self, video_fn, list_images):
        out = cv2.VideoWriter(video_fn, cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 256))

        for i in range(len(list_images)):
            out.write(list_images[i])

        out.release()

    def visualize(self, color_image):
        from libs.nn.tps_model import TPS
        from libs.utils.flow import imgshow
        import cv2

        color_image = cv2.resize(color_image, dsize=(256, 256))

        augment_color_image, G = TPS().augment(color_image)

        G = G[np.newaxis, : ,: ,:]
        color_image = color_image[np.newaxis, :, :, :]
        augment_color_image = augment_color_image[np.newaxis, :, :, :]

        grid_process = GridProcessing(G=G, r=self.r)

        B = 1
        F_H = self.feature_h
        F_W = self.feature_w

        list_image = []

        anchor_cords, positive_cords, negative_cords = grid_process._prepare()
        for anchor_cord, positive_cord, negative_cord in zip(anchor_cords, positive_cords, negative_cords):
            b = anchor_cord[0]

            _color = color_image[b].copy()
            _ref = augment_color_image[b].copy()

            cv2.circle(_color, (int(anchor_cord[1] * self.r), int(anchor_cord[2] * self.r)), radius=8,
                       color=(0, 0, 255), thickness=3)
            cv2.circle(_ref, (int(positive_cord[1] * self.r), int(positive_cord[2] * self.r)), radius=8,
                       color=(0, 255, 0), thickness=3)
            cv2.circle(_ref, (int(negative_cord[1] * self.r), int(negative_cord[2] * self.r)), radius=3,
                       color=(255, 255, 0), thickness=2)

            list_image += [np.concatenate([_ref, _color], axis=1)]

        return list_image

class GridProcessing():
    def __init__(self, G, r):
        self.G = G #G.cpu().numpy()
        self.r = r
        self.min_threshold = 0.01

        self.G_norm = self._decompose_G(self.G)

    def _decompose_G(self, G):
        B, H, W, _ = G.shape

        G_xmin = np.floor(G[:, :, :, [0]] * W).astype(np.int32)
        G_xmax = G_xmin + 1
        G_ymin = np.floor(G[:, :, :, [1]] * H).astype(np.int32)
        G_ymax = G_ymin + 1

        G_norm = np.concatenate([G_xmin, G_ymin, G_xmax, G_ymax], axis=-1)
        return G_norm

    def _prepare(self):
        B, H, W, _ = self.G.shape
        F_H = H // self.r
        F_W = W // self.r

        final_anchor_cords, final_positive_cords, final_negative_cords = [], [], []

        for b in range(0, B):
            for h in range(0, F_H):
                for w in range(0, F_W):
                    #
                    anchor_cords = self._get_positive_ids(b, h, w)
                    if len(anchor_cords) < 1:
                        continue

                    positive_cords = [(w, h)] * len(anchor_cords)
                    negative_cords = self._get_negative_ids(positive_cords)

                    #
                    for anchor_cord, positive_cord, negative_cord in zip(anchor_cords, positive_cords, negative_cords):
                        final_anchor_cords += [[b] + anchor_cord.tolist()]
                        final_positive_cords += [[b] + list(positive_cord)]
                        final_negative_cords += [[b] + list(negative_cord)]

        return final_anchor_cords, final_positive_cords, final_negative_cords

    def _get_negative_ids(self, positive_cords):
        B, H, W, _ = self.G.shape

        negative_cords = []
        for positive_cord in positive_cords:
            # generate negatives
            iys = random.choices(list(range(0, H // self.r)), k=10)
            ixs = random.choices(list(range(0, W // self.r)), k=10)

            #
            tmp_negative_cords = []
            for ix, iy in zip(ixs, iys):
                if (ix, iy) in [positive_cord]: continue
                tmp_negative_cords += [(ix, iy)]

            #
            negative_cord = list(sorted(tmp_negative_cords,
                               key=lambda elem: (elem[1] - positive_cord[1]) ** 2 + (elem[0] - positive_cord[0]) ** 2))[-1]

            #
            negative_cords += [negative_cord]

        return negative_cords


    def _get_positive_ids(self, b, iy, ix):
        # iy, ix from feature map
        B, H, W, _ = self.G.shape

        #
        src_iy_mid = int((iy + 0.5) * self.r)
        src_ix_mid = int((ix + 0.5) * self.r)

        #
        src_G_norm = self.G_norm[b, src_iy_mid, src_ix_mid]
        src_x_min, src_y_min, src_x_max, src_y_max = src_G_norm

        if src_x_min < 0 or src_y_min < 0 or src_x_max > W or src_y_max > H:
            return []

        ix_range = [src_x_min // self.r , src_x_max // self.r + 1]
        iy_range = [src_y_min // self.r, src_y_max // self.r + 1]

        positive_cords = np.array([[(x, y) for y in range(iy_range[0], iy_range[1]) if 0 <= y <= H // self.r]
                                   for x in range(ix_range[0], ix_range[1]) if 0 <= x <= W // self.r])
        positive_cords = positive_cords.reshape((-1, 2)).clip(0, 31)

        return positive_cords


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_cosine(self, x1, x2):
        return torch.sum(F.cosine_similarity(x1, x2))

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def calc_dotproduct(self, x1, x2):
        return torch.sum(torch.matmul(x1, x2.t()))

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = 1 - self.calc_cosine(anchor, positive) #self.calc_euclidean(anchor, positive)
        distance_negative = 1 - self.calc_cosine(anchor, negative) #self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

def get_negative_ids(anchor_f, ref_fs):
    """
    :param anchor_f: (1, n_dim)
    :param ref_fs: (hw, n_dim)
    :return:
    """
    c, h, w = ref_fs.size()
    ref_fs = ref_fs.view(c, h * w).contiguous().permute(1, 0) # (hw, n_dim)

    dist = F.cosine_similarity(anchor_f, ref_fs)

    min_id = torch.argsort(dist).cpu().numpy()[:3]
    return -1, min_id % w, min_id // w

class SimilarityTripletLoss(nn.Module):
    def __init__(self, receptive_field=8, n_positive=2, k=2):
        super(SimilarityTripletLoss, self).__init__()
        self.receptive_field = receptive_field

        self.loss = TripletLoss(margin=0.6)

    def forward(self, sketch_query_vectors, ref_key_vectors, G):
        """
        :param sketch_query_vectors: of size (b, hw, n_dim)
        :param ref_key_vectors: of size (b, hw, n_dim)
        :param G: (b, h, w, 2)
        :return:
        """
        grid_process= GridProcessing(G.cpu().numpy(), r = 8)
        B, H, W, _ = G.size()
        B, C, F_H, F_W = sketch_query_vectors.size()

        assert H == F_H * self.receptive_field and W == F_W * self.receptive_field

        i = 0
        triplet_loss = 0
        anchor_cords, positive_cords, negative_cords = grid_process._prepare()

        for anchor_cord, positive_cord, negative_cord in zip(anchor_cords, positive_cords, negative_cords):
            b = anchor_cord[0]

            anchor = sketch_query_vectors[b, :, [anchor_cord[2]], [anchor_cord[1]]]
            positive = ref_key_vectors[b, :, [positive_cord[2]], [positive_cord[1]]]

            negative_cord = get_negative_ids(anchor.T, ref_key_vectors[b])
            negative = ref_key_vectors[b, :, negative_cord[2], negative_cord[1]]

            _loss = self.loss(anchor=anchor.T.contiguous(), positive=positive.T.contiguous(), negative=negative.T.contiguous())
            triplet_loss += _loss

            i += 1

        return triplet_loss / (1e-6 + i)

if __name__ == '__main__':
    import cv2

    im_fn = "/home/kan/Desktop/Cinnamon/gan/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions/geek/full_data/hor01_sample/color/hor01_018_021_k_A_A0002.png"
    visualize = GridVisualize(org_size=(256,256), feature_size=(32,32))
    list_image = visualize.visualize(color_image=cv2.imread(im_fn))
    visualize.to_video("./output.avi", list_image)

