import click
import torch
import os
import yaml
from copy import deepcopy
from torch.utils import data
import torchvision.utils as vutils
import torch.optim as optim

from libs.utils.flow import to_device, create_train_transform
from libs.loss.gan_loss import dis_loss, gen_loss
from libs.loss.content_style_loss import L1StyleContentLoss
from libs.loss.triplet_loss import SimilarityTripletLoss
from libs.nn.color_model import ColorModel
from libs.nn.gan_model import Discriminator

from libs.dataset.anime_dataset import OneImageAnimeDataset as AnimeDataset

@click.command()
@click.option('--cfg', default='./exps/_simple.yaml', help='Path to Config Path')
def main(cfg):
    cfg = yaml.load(open(cfg, 'r'))

    # dataset
    _cfg = cfg['TRAIN']
    train_dataset = AnimeDataset(input_dir=_cfg['INPUT_DIR'], is_train=True, transform=create_train_transform(cfg))
    train_loader  = data.DataLoader(train_dataset, batch_size=_cfg['BATCH_SIZE'], shuffle=True, num_workers=_cfg['N_WORK'])

    # model
    color_model = ColorModel(attn_in_dim=256)
    disc_model  = Discriminator(ch_input=6)

    color_model.cuda()
    disc_model.cuda()

    # loss
    _cfg = cfg['LOSSES']
    l1_style_content_loss = L1StyleContentLoss(content_layers=_cfg['CONTENT']['LAYER'], style_layers=_cfg['STYLE']['LAYER']).cuda()
    sim_triplet_loss = SimilarityTripletLoss(n_positive=_cfg['TRIPLET']['N_POSITIVE'], k=_cfg['TRIPLET']['K']).cuda()
    loss_weights = {} #{k:v['WEIGHT'] for k,v in _cfg.items()}

    # optimizer
    color_optim = optim.Adam(color_model.parameters(), lr=cfg['TRAIN']['G_LR'], betas=(0, 0.999))
    disc_optim = optim.Adam(disc_model.parameters(), lr=cfg['TRAIN']['D_LR'], betas=(0, 0.999))

    os.makedirs('samples', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    for epoch_id in range(cfg['TRAIN']['EPOCH']):
        print ('>> Epoch:', epoch_id + 1)

        train(train_loader, (color_model, disc_model), (l1_style_content_loss, sim_triplet_loss),
              loss_weights, (color_optim, disc_optim), 'samples', 'weights')

global_iter = 0
def train(loader, models, losses, loss_weights, optimizers, vis_dir, weight_dir):
    global global_iter

    # models
    color_model, disc_model = models
    color_model.train()
    disc_model.train()

    # optimizer
    color_optimzier, disc_optimizer = optimizers

    # losses
    l1_style_content_loss, sim_triplet_loss = losses
    lws = {
        'triplet': 1.,
        'l1': 30.,
        'gan_gen': 1.,
        'style': 100.,
        'content': 0.01
    }

    for batch_id, batch_info in enumerate(loader):
        s_im, ref_im, ref_augment_im, meta = batch_info
        s_im = to_device(s_im)
        ref_im = to_device(ref_im)
        ref_augment_im = to_device(ref_augment_im)

        # visualize
        vutils.save_image(s_im.detach().cpu(), filename='vis_debug_im/input.sketch.png', normalize=True, range=(0,1), padding=0)
        vutils.save_image(ref_im, filename='vis_debug_im/input.ref.png', normalize=True, range=(0,1), padding=0)
        vutils.save_image(ref_augment_im, filename='vis_debug_im/input.ref_augment.png', normalize=True, range=(0,1), padding=0)

        # predict
        G = meta['G']
        o_im, sketch_f, refer_f = color_model(s_im, ref_augment_im)

        # loss
        ### >> for discriminator
        input_real = torch.cat([ref_im, s_im], dim=1)
        input_fake = torch.cat([o_im, s_im], dim=1)

        disc_optimizer.zero_grad()
        gan_dis_score = dis_loss(disc_model, input_real, input_fake)
        gan_dis_score.backward()
        disc_optimizer.step()

        ### >> for style, content, l1
        color_optimzier.zero_grad()
        style_score, content_score, l1_score = l1_style_content_loss(o_im, ref_im)

        ### >> for attention & gan generator
        sim_triplet_score = sim_triplet_loss(sketch_f, refer_f, G)
        gan_gen_score = gen_loss(disc_model, input_fake)

        total_loss = lws['style'] * style_score + lws['content'] * content_score + lws['l1'] * l1_score + \
                     lws['triplet'] * sim_triplet_score + lws['gan_gen'] * gan_gen_score

        total_loss.backward()
        color_optimzier.step()

        global_iter += 1
        print('\t[Iter]: %d, [Style]:%.3f, [Content]:%.3f, [L1]:%.3f, [TRIPLET]:%.3f, [GAN_G]:%.3f, [GAN_D]:%.3f' %
              (global_iter, style_score.item() * lws['style'], content_score.item() * lws['content'], l1_score.item() * lws['l1'], sim_triplet_score.item() * lws['triplet'],
               gan_gen_score.item() * lws['gan_gen'], gan_dis_score.item())
        )


        if global_iter % 1 == 0:
            """
            >> Save weights
            """
            save_G_path = os.path.join(weight_dir, '{:08d}.G.pth'.format(global_iter))
            save_D_path = os.path.join(weight_dir, '{:08d}.D.pth'.format(global_iter))
            save_optimG_path = os.path.join(weight_dir, '{:08d}.optimG.pth'.format(global_iter))
            save_optimD_path = os.path.join(weight_dir, '{:08d}.optimD.pth'.format(global_iter))

            torch.save(color_model.state_dict(), save_G_path)
            torch.save(disc_model.state_dict(), save_D_path)
            torch.save(color_optimzier.state_dict(), save_optimG_path)
            torch.save(disc_optimizer.state_dict(), save_optimD_path)

            """
            >> Visualize 
            """
            color_model.eval()

            with torch.no_grad():
                o_im, _, _ = color_model(s_im, ref_augment_im)

            save_output_path = os.path.join(vis_dir, '{:08d}.output.png'.format(global_iter))
            save_sketch_path = os.path.join(vis_dir, '{:08d}.sketch.png'.format(global_iter))
            save_ref_augment_path = os.path.join(vis_dir, '{:08d}.ref_augment.png'.format(global_iter))
            save_target_path = os.path.join(vis_dir, '{:08d}.target.png'.format(global_iter))

            vutils.save_image(o_im, save_output_path, normalize=True, range=(0,1))
            vutils.save_image(s_im, save_sketch_path, normalize=True, range=(0, 1))
            vutils.save_image(ref_augment_im, save_ref_augment_path, normalize=True, range=(0, 1))
            vutils.save_image(ref_im, save_target_path, normalize=True, range=(0, 1))


if __name__ == '__main__':
    main()