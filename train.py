import click
import torch
import os
import yaml

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from libs.utils.flow import create_train_transform, save_model, save_image_local, save_image_tensorboard
from libs.loss.gan_loss import dis_loss, gen_loss
from libs.loss.content_style_loss import L1StyleContentLoss
from libs.loss.triplet_loss import SimilarityTripletLoss
from libs.loss.semantic_loss import semantic_loss_fn
from libs.nn.color_model import ColorModel, Discriminator
from libs.dataset.anime_dataset import OneImageAnimeDataset as AnimeDataset

######
"""
global parameters
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_iter = 0

def ensure_loss(loss):
    assert loss.requires_grad == True, 'Loss without gradient'

### end ###

@click.command()
@click.option('--cfg', default='./exps/_simple.yaml', help='Path to Config Path')
def main(cfg):
    cfg = yaml.load(open(cfg, 'r'))
    train_cfg = cfg['TRAIN']
    model_cfg = cfg['MODELS']

    # writer
    writer= SummaryWriter(log_dir='color')
    vis_dir, weight_dir = 'samples', 'weights'

    # dataset
    train_dataset = AnimeDataset(input_dir=train_cfg['INPUT_DIR'], feat_size=(16,16), im_size=(256,256))
    train_loader  = data.DataLoader(train_dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, num_workers=train_cfg['N_WORK'])

    # model
    color_model = ColorModel()
    disc_model  = Discriminator(ch_input=6)

    color_model.to(device)
    disc_model.to(device)

    # loss
    l1_style_content_loss = L1StyleContentLoss().to(device)
    sim_triplet_loss = None #SimilarityTripletLoss().to(device)
    semantic_loss = semantic_loss_fn().to(device)
    loss_weights = {}

    # optimizer
    color_optim = optim.Adam(color_model.parameters(), lr=train_cfg['G_LR'], betas=(0.5, 0.999))
    disc_optim  = optim.Adam(disc_model.parameters(), lr=train_cfg['D_LR'], betas=(0.5, 0.999))

    # lr scheduler
    color_scheduler = optim.lr_scheduler.StepLR(color_optim, step_size=1000, gamma=0.1)
    disc_scheduler  = optim.lr_scheduler.StepLR(disc_optim, step_size=1000, gamma=0.1)

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    for epoch_id in range(cfg['TRAIN']['EPOCH']):
        print ('>> Epoch:', epoch_id + 1)

        train(train_loader,
              (color_model, disc_model), #models
              (l1_style_content_loss, sim_triplet_loss, semantic_loss), #losses
              loss_weights, #loss_weights
              (color_optim, disc_optim), #optimizer
              vis_dir, weight_dir, writer, cfg)  # other stuffs

        color_scheduler.step()
        disc_scheduler.step()

    writer.close()


def train(loader, models, losses, loss_weights, optimizers, vis_dir, weight_dir, writer, cfg):
    global global_iter

    # models
    color_model, disc_model = models
    color_model.train()
    color_model.feature_extraction.eval()
    disc_model.train()

    # optimizer
    color_optimzier, disc_optimizer = optimizers

    # losses & weights
    l1_style_content_loss, sim_triplet_loss, semantic_loss = losses
    loss_weight_cfg = cfg['LOSSES']['WEIGHTS']
    lws = {
        'l1': loss_weight_cfg['L1'],
        'gan_gen': loss_weight_cfg['GAN_G'],
        'style': loss_weight_cfg['STYLE'],
        'content': loss_weight_cfg['CONTENT'],
        'sm': 1.
    }

    for batch_id, batch_info in enumerate(loader):
        #
        list_ref, list_tgt = batch_info
        sketch_ref, color_ref, mask_ref = list_ref
        sketch_tgt, color_tgt, mask_tgt = list_tgt

        #
        sketch_ref  = sketch_ref.to(device)
        sketch_tgt  = sketch_tgt.to(device)
        color_ref   = color_ref.to(device)
        color_tgt   = color_tgt.to(device)
        mask_ref    = mask_ref.to(device)
        mask_tgt    = mask_tgt.to(device)

        #
        output, semantic_output = color_model(color_ref, sketch_ref, sketch_tgt, 'train',
                                              GT_src_mask=mask_ref, GT_tgt_mask=mask_tgt)

        # training discriminator
        input_real = torch.cat([color_tgt, sketch_tgt], dim=1)
        input_fake = torch.cat([output, sketch_tgt], dim=1)

        disc_optimizer.zero_grad()
        g_d_loss = dis_loss(disc_model, input_real, input_fake)
        g_d_loss.backward()
        disc_optimizer.step()

        # training generator
        color_optimzier.zero_grad()

        #1
        g_g_loss = gen_loss(disc_model, input_fake)

        #2
        sm_loss, l1_sm, l2_sm, l3_sm = semantic_loss(semantic_output, mask_ref, mask_tgt)

        #3
        style_loss, content_loss, l1_loss = l1_style_content_loss(output, color_tgt)

        #all
        style_loss *= lws['style']
        content_loss *= lws['content']
        l1_loss *= lws['l1']
        g_g_loss *= lws['gan_gen']
        sm_loss *= lws['sm']

        total_loss = style_loss + content_loss + l1_loss + g_g_loss + sm_loss
        total_loss.backward()

        color_optimzier.step()

        global_iter += 1
        print('\t[iter]: %d, [style]:%.3f, [content]:%.3f, [l1]:%.3f, [G_g]:%.3f, [G_d]:%.3f, [sm]:%.3f' %
              (global_iter, style_loss.item() , content_loss.item(), l1_loss.item(), g_g_loss.item(),
               g_d_loss.item(), sm_loss.item())
        )

        if global_iter % 10 == 0:
            """
            >> save weights
            """
            save_model(global_iter, weight_dir, color_model, disc_model, color_optimzier, disc_optimizer)

            """
            >> visualize 
            """
            color_model.eval()

            with torch.no_grad():
                output, _ = color_model(color_ref, sketch_ref, sketch_tgt, '!train')

            color_model.train()

            save_image_local(global_iter, writer,
                             style_loss, content_loss, l1_loss, sm_loss, g_g_loss, g_d_loss,
                             output, sketch_tgt, color_ref, color_tgt)

            """
            tensor-board
            """
            save_image_tensorboard(global_iter, vis_dir, output, sketch_tgt, color_ref, color_tgt)

if __name__ == '__main__':
    main()