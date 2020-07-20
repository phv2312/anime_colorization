import torch
import random
import numpy as np
import torch.nn as nn
from math import floor

def _calc_dot_product(elem1, elem2):
    return torch.matmul(elem1, elem2.t())

def _to_ys(idx):
    return [_[1] for _ in idx]

def _to_xs(idx):
    return [_[0] for _ in idx]

class GridProcessing():
    def __init__(self, G, r):
        self.G = G.cpu().numpy()
        self.r = r
        self.min_threshold = 0.01

        self.G_norm = self._decompose_G(self.G)

    def _decompose_G(self, G):
        B, H, W, _ = G.shape

        G_xmin = np.floor(G[:,:,:,[0]] * W).astype(np.int32)
        G_xmax = G_xmin + 1
        G_ymin = np.floor(G[:,:,:,[1]] * H).astype(np.int32)
        G_ymax = G_ymin + 1

        G_norm = np.concatenate([G_xmin, G_ymin, G_xmax, G_ymax], axis=-1)
        return G_norm

    def get_negative_ids(self, positive_cords):
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


    def get_positive_ids(self, b, iy, ix):
        B, H, W, _ = self.G.shape

        #
        src_iy_mid = int((iy + 0.5) * self.r)
        src_ix_mid = int((ix + 0.5) * self.r)

        #
        src_G_norm = self.G_norm[b, src_iy_mid, src_ix_mid]
        src_x_min, src_y_min, src_x_max, src_y_max = src_G_norm

        if src_x_min < 0 or src_y_min < 0 or src_x_max > W // self.r or src_y_max > H // self.r:
            return []

        ix_range = [src_x_min // self.r , src_x_max // self.r + 1]
        iy_range = [src_y_min // self.r, src_y_max // self.r + 1]

        positive_cords = np.array([[(x, y) for y in range(iy_range[0], iy_range[1]) if 0 <= y <= H//self.r]
                                   for x in range(ix_range[0], ix_range[1]) if 0 <= x <= W//self.r])
        positive_cords = positive_cords.reshape((-1, 2))

        return positive_cords


class TripletLoss(nn.Module):
    def __init__(self, margin=12.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class SimilarityTripletLoss(nn.Module):
    def __init__(self, receptive_field=8, n_positive=2, k=2):
        super(SimilarityTripletLoss, self).__init__()
        self.receptive_field = receptive_field

        self.loss = TripletLoss(margin=12.) #nn.TripletMarginLoss(margin=12.)

    def forward(self, sketch_query_vectors, ref_key_vectors, G):
        """
        :param sketch_query_vectors: of size (b, hw, n_dim)
        :param ref_key_vectors: of size (b, hw, n_dim)
        :param G: (b, h, w, 2)
        :return:
        """
        grid_process= GridProcessing(G, r = 8)
        B, H, W, _ = G.size()
        B, C, F_H, F_W = sketch_query_vectors.size()

        assert H == F_H * self.receptive_field and W == F_W * self.receptive_field

        i = 0
        triplet_loss = 0
        # iterate
        for b in range(0, B):
            for h in range(0, F_H):
                for w in range(0, F_W):
                    #
                    anchor_cords = grid_process.get_positive_ids(b, h, w) #GridProcessing(G).get_pos_ids(b, h * 8, w * 8)
                    if len(anchor_cords) < 1: continue

                    positive_cords = [(h, w)] * len(anchor_cords)
                    negative_cords = grid_process.get_negative_ids(positive_cords)

                    #
                    for anchor_cord, positive_cord, negative_cord in zip(anchor_cords, positive_cords, negative_cords):
                        anchor = sketch_query_vectors[b, :, [anchor_cord[1]], [anchor_cord[0]]]
                        positive = ref_key_vectors[b, :, [positive_cord[1]], [positive_cord[0]]]
                        negative = ref_key_vectors[b, :, [negative_cord[1]], [negative_cord[0]]]

                        _loss = self.loss(anchor=anchor.T, positive=positive.T, negative=negative.T)
                        triplet_loss += _loss
                        i += 1

        return triplet_loss / (1e-6 + i)


