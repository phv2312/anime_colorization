import torch
import click
import numpy as np
import cv2

from torchvision.transforms import transforms
from scipy.spatial.distance import cdist

from libs.nn.tps_model import TPS
from libs.utils.flow import load_model, imgshow
from libs.utils.dataset import read_im
from libs.nn.color_model import ColorModel, Discriminator

def gen_reference_image(colored_image):
    augmented_image, _, _ = TPS().augment(colored_image)

    return augmented_image

def get_attention(sketch_queries_f, refer_key_f):
    """

    :param sketch_queries_f: shape (1, dim, 32, 32)  == (b,c,h,w)
    :param refer_key_f: shape (1, dim, 32, 32)
    :return:
    """

    sketch_queries_f = sketch_queries_f.squeeze(0).transpose(1,2,0) # (h,w,d)
    refer_key_f = refer_key_f.squeeze(0).transpose(1,2,0) # (h,w,d)

    h, w, d = sketch_queries_f.shape

    sketch_queries_f = sketch_queries_f.reshape((h * w, d))
    refer_key_f = refer_key_f.reshape((h * w, d))

    matrix_dist = cdist(sketch_queries_f, refer_key_f, metric='cosine')
    min_refer_ids = np.argmin(matrix_dist, axis=-1)

    # #
    # _min_refer_ids = np.argsort(matrix_dist, axis=-1)
    # for sketch_id in range(len(matrix_dist)):
    #     _matrix_dist = _min_refer_ids[sketch_id][:5]
    #
    #     s_h, s_w = sketch_id // w, sketch_id % w
    #
    #     r_hs = _matrix_dist // w
    #     r_ws = _matrix_dist % w
    #
    #     print ('Source:', (s_h, s_w))
    #
    #     print ('Tagt:', (s_h, s_w), ',dist:', matrix_dist[sketch_id, sketch_id])
    #     print (matrix_dist[sketch_id].mean(), matrix_dist[sketch_id].max())
    #
    #     for _id, (r_h, r_w) in enumerate(zip(r_hs, r_ws)):
    #         print ('Pred:', (r_h, r_w), ',dist:', matrix_dist[sketch_id, _matrix_dist[_id]])
    #
    #
    #     input('press enter to continue ...')
    #
    #
    pair = {}



    for sketch_id, refer_id in enumerate(min_refer_ids):


        s_h, s_w = sketch_id // w, sketch_id % w
        r_h, r_w = refer_id // w, refer_id % w

        pair[(s_w, s_h)] = (r_w, r_h)

        print((s_h, s_w), (r_h, r_w), matrix_dist[sketch_id][refer_id])

    return pair

def to_video(video_fn, list_images):
    out = cv2.VideoWriter(video_fn, cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 256))

    for i in range(len(list_images)):
        out.write(list_images[i])

    out.release()

# @click.command()
# @click.option('--cfg', default='./exps/_simple.yaml', help='Path to Config Path')
# @click.option('--weight_path', default="", help="Path to weight")
def main(cfg, weight_path, sketch_path, reference_path):
    # prepare model & transforms
    color_model = load_model(ColorModel(), weight_path).eval()
    color_model.cuda()

    infer_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5, ))
    ])

    # prepare input
    reference_image = gen_reference_image(read_im(reference_path))
    sketch_image = read_im(sketch_path)

    reference_input = infer_transforms(reference_image).unsqueeze(0).cuda()
    sketch_input = infer_transforms(sketch_image)[:1,:,:].unsqueeze(0).cuda()

    # infer ....
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    with torch.no_grad():
        o_im, sketch_queries_f, refer_key_f = color_model(reference_input, sketch_input)

    o_im = o_im.squeeze(0).cpu()
    r_im = reference_input.squeeze(0).cpu()

    o_im = o_im * torch.tensor(data=std).view(3, 1, 1) + torch.tensor(data=mean).view(3, 1, 1)
    o_im = transforms.ToPILImage(mode='RGB')(o_im)
    r_im = r_im * torch.tensor(data=std).view(3, 1, 1) + torch.tensor(data=mean).view(3, 1, 1)
    r_im = transforms.ToPILImage(mode='RGB')(r_im)

    o_im = np.array(o_im)
    r_im = np.array(r_im)

    # visualize attention
    pair = get_attention(sketch_queries_f.cpu().numpy(), refer_key_f.cpu().numpy())
    r = 16
    list_image = []
    for (src_w, src_h), (tgt_w, tgt_h) in pair.items():
        print ((src_w, src_h), (tgt_w, tgt_h))

        _o_im = o_im.copy()
        _r_im = r_im.copy()

        cv2.circle(_o_im, (int(r * src_w), int(r * src_h)), radius=3, color=(255, 0, 0), thickness=2)
        cv2.circle(_r_im, (int(r * tgt_w), int(r * tgt_h)), radius=3, color=(255, 0, 0), thickness=2)

        list_image += [np.concatenate([_r_im, _o_im], axis=1)]

    to_video("./debug_vis_attn.avi", list_image)

if __name__ == '__main__':
    sketch_path = "/home/kan/Desktop/cinnamon/anime_colorization/simple_data/hor01_sample/sketch/hor01_018_021_k_A_A0002.png"
    refer_path = "/home/kan/Desktop/cinnamon/anime_colorization/simple_data/hor01_sample/color/hor01_018_021_k_A_A0001.png"
    weight_path = "/home/kan/Desktop/cinnamon/anime_colorization/weights/00000630.G.pth"

    main(None, weight_path, sketch_path, refer_path)