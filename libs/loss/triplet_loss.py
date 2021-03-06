import torch
import random
import numpy as np
import torch.nn as nn
import copy
import torch.nn.functional as F

from libs.nn.thinplate import numpy as tps

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
        out = cv2.VideoWriter(video_fn, cv2.VideoWriter_fourcc(*'DIVX'), 15, (self.org_w * 2, self.org_h))

        for i in range(len(list_images)):
            out.write(list_images[i])

        out.release()

    def visualize(self, color_image):
        from libs.nn.tps_model import TPS
        import cv2

        color_image = cv2.resize(color_image, dsize=(self.org_w, self.org_h))

        augment_color_image, theta, c_dist = TPS().augment(color_image)

        theta = theta[np.newaxis, :, :] # dummy batch size
        c_dist = c_dist[np.newaxis, :, :] # dummy batch size
        color_image = color_image[np.newaxis, :, :, :]
        augment_color_image = augment_color_image[np.newaxis, :, :, :]

        grid_process = GridProcessing(theta, c_dist, (self.feature_h, self.feature_w))
        positive_cords = grid_process.get_positive()

        list_image = []
        for (ac_b, ac_iy, ac_ix), (po_b, po_iy, po_ix) in positive_cords.items():
            _color_image = color_image[ac_b].copy()
            _augment_color_image = augment_color_image[po_b].copy()

            cv2.circle(_color_image, (int(ac_ix * self.r), int(ac_iy * self.r)), radius=8,
                       color=(0, 0, 255), thickness=3)
            cv2.circle(_augment_color_image, (int(po_ix * self.r), int(po_iy * self.r)), radius=8,
                       color=(0, 255, 0), thickness=3)

            list_image += [np.concatenate([_augment_color_image, _color_image], axis=1)]

        return list_image

class GridProcessing():
    def __init__(self, theta=None, c_dst=[], f_shape=(32, 32)):
        self.f_shape = f_shape

        self.G = tps.batch_tps_grid(theta, c_dst, dshape=f_shape) # (b,h,w,2~xy)

        self.G_new = self._decompose_G(self.G)

    def _decompose_G(self, G):
        B, H, W, _ = G.shape

        G *= np.array([W, H]).reshape((1, 1, 1, 2))

        G_xmin = np.floor(G[:, :, :, [0]]).astype(np.int32)
        G_xmax = G_xmin + 1
        G_ymin = np.floor(G[:, :, :, [1]]).astype(np.int32)
        G_ymax = G_ymin + 1

        """
        1 +++ 2
        +++++++
        4 +++ 3
        """

        G1 = np.concatenate([G_xmin, G_ymin], axis=-1)
        G2 = np.concatenate([G_xmax, G_ymin], axis=-1)
        G3 = np.concatenate([G_xmax, G_ymax], axis=-1)
        G4 = np.concatenate([G_xmin, G_ymax], axis=-1)

        G1_dist = np.linalg.norm(G - G1, ord=2, axis=-1)
        G2_dist = np.linalg.norm(G - G2, ord=2, axis=-1)
        G3_dist = np.linalg.norm(G - G3, ord=2, axis=-1)
        G4_dist = np.linalg.norm(G - G4, ord=2, axis=-1)

        G1 = np.concatenate([G1, G1_dist[:,:,:,np.newaxis]], axis=-1)
        G2 = np.concatenate([G2, G2_dist[:,:,:,np.newaxis]], axis=-1)
        G3 = np.concatenate([G3, G3_dist[:,:,:,np.newaxis]], axis=-1)
        G4 = np.concatenate([G4, G4_dist[:,:,:,np.newaxis]], axis=-1)

        G_new = np.stack([G1, G2, G3, G4], axis=3)
        return G_new

    def get_positive(self):
        B, H, W, _, _ = self.G_new.shape

        positive_pair = {} # key from sketch-anchor, value from reference
        for b in range(B):
            _G_new = self.G_new[b] # (H, W, 4, 3)

            # get_positive
            for h in range(H):
                for w in range(W):

                    p1, p2, p3, p4 = _G_new[h, w, :, :2]
                    p1_dist, p2_dist, p3_dist, p4_dist = _G_new[h, w, : ,-1]
                    ps = []
                    dists = []

                    # filter
                    for p_id, (p, dist) in enumerate(zip([p1, p2, p3, p4], [p1_dist, p2_dist, p3_dist, p4_dist])):
                        if p[0] < 0 or p[1] < 0: continue #lower bound
                        if p[0] > W - 1 or p[1] > H - 1: continue #upper bound
                        ps += [p]; dists += [dist]

                    if len(ps) > 0:

                        dists = np.array(dists)
                        probs = np.max(dists) - dists + 0.3
                        probs = probs / np.sum(probs)

                        p_pos_id = int(np.argmax(probs)) #np.random.choice(np.arange(len(ps)), p=probs)
                        positive_pair[(b, int(ps[p_pos_id][1]), int(ps[p_pos_id][0]))] = (b, int(h), int(w))

        return positive_pair

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_cosine(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2)

    def forward(self, anchor, positive, negative, print_value=False) -> torch.Tensor:
        distance_positive = self.calc_cosine(anchor, positive)
        distance_negative = self.calc_cosine(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        if print_value:
            print ('pos:', distance_positive)
            print ('neg:', distance_negative)

        return losses

def calculate_cosine_distance(sketch_f, ref_fs):
    """
    sketch_f: of shape (D, 1),
    ref_fs: of shape (D, F_H F_W)
    """
    D, F_H, F_W = ref_fs.shape
    ref_fs_flat = ref_fs.view(D, F_H * F_W)

    sim = F.cosine_similarity(sketch_f.T, ref_fs_flat.T)
    dis = torch.ones_like(sim) - sim

    return dis.view(F_H, F_W)

class SimilarityTripletLoss(nn.Module):
    def __init__(self):
        super(SimilarityTripletLoss, self).__init__()

    def forward(self, sketch_query_vectors, ref_key_vectors, theta, c_dst):
        """
        :param sketch_query_vectors: of size (b, hw, n_dim)
        :param ref_key_vectors: of size (b, hw, n_dim)
        :param ...
        :return:
        """
        B, C, F_H, F_W = sketch_query_vectors.size()
        grid_process = GridProcessing(theta.cpu().numpy(), c_dst.cpu().numpy(), (F_H, F_W))

        i = 0
        triplet_loss = 0
        positive_cords = grid_process.get_positive()

        pos_distances, neg_distances = [], []
        for (ac_b, ac_iy, ac_ix), (po_b, po_iy, po_ix) in positive_cords.items():
            sketch_ac_f = sketch_query_vectors[ac_b, :, [ac_iy], [ac_ix]]
            ref_po_f = ref_key_vectors[po_b, :, [po_iy], [po_ix]]

            # get negative
            distance = calculate_cosine_distance(sketch_ac_f, ref_key_vectors[po_b])
            ignore_pos = torch.zeros_like(distance)
            ignore_pos[po_iy, po_ix] = 2. # ignore positive

            min_distances = torch.topk(distance.flatten() + ignore_pos.flatten(), k=4, largest=False, dim=-1)[0]
            pos_distance  = distance[po_iy, po_ix]

            neg_distances.append(min_distances)
            pos_distances.append(pos_distance)

        pos_distance = torch.mean(torch.stack(pos_distances, dim=-1))
        loss = pos_distance

        if len(neg_distances) > 0:
            neg_distances = torch.cat(neg_distances, dim=-1)
            neg_loss = torch.clamp(torch.full_like(neg_distances, 0.6) - neg_distances, min=0.0)
            neg_loss = torch.mean(neg_loss)
            loss = loss + neg_loss
        return loss


if __name__ == '__main__':
    import cv2

    im_fn = "/home/kan/Desktop/Cinnamon/gan/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions/geek/full_data/hor01_sample/color/hor01_018_021_k_A_A0002.png"
    visualize = GridVisualize(org_size=(256,256), feature_size=(16, 16))
    list_image = visualize.visualize(color_image=cv2.imread(im_fn))
    visualize.to_video("./output.avi", list_image)

