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

class GridProcessing():
    def __init__(self, G, n_positive, k):
        self.G = G
        self.n_positive = n_positive
        self.k = k

    def get_pos_negative_ids(self, b, y, x, receptive_field = 8):
        """
        :param b: idx
        :param y: idx
        :param x: idx
        :return:
        """

        B, H, W, _ = self.G.size()
        n_positive = self.n_positive
        k = self.k

        positive_ids = []
        negative_ids = []

        ix, iy = self.G[b, y, x]

        # bot-left
        ix_nw = int(floor(ix))
        iy_nw = int(floor(iy))

        # top-right
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

        # conduct positive ids
        for _x in range(ix_nw, ix_se + 1):
            for _y in range(iy_nw, iy_se + 1):

                if 0 <= _x <= W and 0 <= _y <= H:
                    f_ix, f_iy = _x // receptive_field, _y // receptive_field
                    if (f_ix, f_iy) not in positive_ids: positive_ids += [(_x, _y)]

        # conduct negative ids
        iys = random.choices(list(range(0, H // receptive_field)), k = 10)
        ixs = random.choices(list(range(0, W // receptive_field)), k = 10)

        for ix, iy in zip(ixs, iys):
            if (ix, iy) in positive_ids: continue
            negative_ids += [(ix, iy)]

        # normalize
        if len(positive_ids) > n_positive:
            positive_ids = list(sorted(positive_ids, key=lambda elem: (elem[1] - y) ** 2 + (elem[0] - x) ** 2 ))
            positive_ids = positive_ids[:n_positive]

        if len(negative_ids) > (n_positive * k):
            negative_ids = list(sorted(negative_ids, key=lambda elem: (elem[1] - y) ** 2 + (elem[0] - x) ** 2 ))[::-1]
            negative_ids = negative_ids[:(n_positive * k)]

        return positive_ids, negative_ids

class SimilarityTripletLoss(nn.Module):
    def __init__(self, receptive_field=8, n_positive=2, k=2):
        super(SimilarityTripletLoss, self).__init__()
        self.receptive_field = receptive_field
        self.n_positive = n_positive
        self.k = 1

        self.loss = TripletLoss(margin=12.) #nn.TripletMarginLoss(margin=12.)

    def forward(self, sketch_context_vectors, ref_context_vectors, G):
        """
        :param sketch_context_vectors: of size (b, hw, n_dim)
        :param ref_context_vectors: of size (b, hw, n_dim)
        :param G: (b, h, w, 2)
        :return:
        """

        #
        B, H, W, _ = G.size()
        featured_H, featured_W = H // self.receptive_field, W // self.receptive_field

        sketch_context = sketch_context_vectors.permute(0,2,3,1)
        ref_context    = ref_context_vectors.permute(0,2,3,1)

        #
        triplet_losses = 0
        i = 0
        for b in range(0, B):
            for h in range(0, featured_H):
                for w in range(0, featured_W):
                    ref_vector = ref_context[b, [h], [w]]
                    p_ids, n_ids = GridProcessing(G, self.n_positive, self.k).get_pos_negative_ids(b,
                                                                                h * self.receptive_field,
                                                                                w * self.receptive_field,
                                                                                self.receptive_field)

                    ske_p_vectors = sketch_context[b, _to_ys(p_ids), _to_xs(p_ids), :]
                    ske_n_vectors = sketch_context[b, _to_ys(n_ids), _to_xs(n_ids), :]

                    if len(ske_p_vectors) > 0 and len(ske_n_vectors) > 0:
                        try:
                            _loss = self.loss(anchor=ref_vector, positive=ske_p_vectors, negative=ske_n_vectors)
                            triplet_losses += _loss
                            i += 1
                        except Exception as e:
                            print ('[Triplet Loss Error T.T]' ,e)

        return triplet_losses / (1e-6 + i)


