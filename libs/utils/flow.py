import torch
import os
from torchvision.transforms import transforms
from torchvision import utils as vutils
import matplotlib.pyplot as plt

def imgshow(im):
    plt.imshow(im)
    plt.show()

def create_train_transform(cfg):
    return transforms.Compose([
        transforms.Resize(size=(cfg['MODELS']['INPUT_HEIGHT'], cfg['MODELS']['INPUT_WEIGHT'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def create_test_transform(cfg):
    return create_train_transform(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_device(x):
    return x.to(device)

def load_model(model, weight_path):
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    return model

from copy import deepcopy
def save_model(global_iter, weight_dir, color_model, disc_model, color_optimizer, disc_optimizer):
    save_G_path = os.path.join(weight_dir, '{:08d}.G.pth'.format(global_iter))
    save_D_path = os.path.join(weight_dir, '{:08d}.D.pth'.format(global_iter))
    save_optimG_path = os.path.join(weight_dir, '{:08d}.optimG.pth'.format(global_iter))
    save_optimD_path = os.path.join(weight_dir, '{:08d}.optimD.pth'.format(global_iter))

    torch.save(color_model.state_dict(), save_G_path)
    torch.save(disc_model.state_dict(), save_D_path)
    torch.save(color_optimizer.state_dict(), save_optimG_path)
    torch.save(disc_optimizer.state_dict(), save_optimD_path)

def save_image_tensorboard(global_iter, vis_dir, o_im, s_im, ref_augment_im, ref_im):
    save_output_path = os.path.join(vis_dir, '{:08d}.output.png'.format(global_iter))
    save_sketch_path = os.path.join(vis_dir, '{:08d}.sketch.png'.format(global_iter))
    save_ref_augment_path = os.path.join(vis_dir, '{:08d}.ref_augment.png'.format(global_iter))
    save_target_path = os.path.join(vis_dir, '{:08d}.target.png'.format(global_iter))

    vutils.save_image(o_im, save_output_path, normalize=True, range=(-1, 1))
    vutils.save_image(s_im, save_sketch_path, normalize=True, range=(-1, 1))
    vutils.save_image(ref_augment_im, save_ref_augment_path, normalize=True, range=(-1, 1))
    vutils.save_image(ref_im, save_target_path, normalize=True, range=(-1, 1))

def save_image_local(global_iter, writer,
                     style_score, content_score, l1_score, sim_triplet_score, gan_gen_score, gan_dis_score,
                     o_im, s_im, ref_augment_im, ref_im):
    for _loss_name, _loss_val in zip(['style', 'content', 'l1', 'triplet', 'gan_g', 'gan_d'],
                                     [style_score, content_score, l1_score, sim_triplet_score, gan_gen_score,
                                      gan_dis_score]):
        writer.add_scalar('train/loss/%s' % _loss_name, _loss_val, global_iter)

    for _image_name, _image_tensor in zip(['output', 'sketch', 'reference', 'target'],
                                          [o_im, s_im, ref_augment_im, ref_im]):
        writer.add_image('train/%s' % _image_name, vutils.make_grid(_image_tensor, normalize=True, range=(-1, 1)),
                         global_iter)

