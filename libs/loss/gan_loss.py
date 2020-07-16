import torch
import torch.nn.functional as F

# hinge loss
def dis_loss(D, real, fake):
    # remember to detach fake
    d_out_real = D(real)
    d_out_fake = D(fake.detach())
    loss_real = F.relu(1.0 - d_out_real).mean()
    loss_fake = F.relu(1.0 + d_out_fake).mean()
    return loss_real + loss_fake

def gen_loss(D, fake):
    d_out = D(fake)
    return -(d_out).mean()
