from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses
import pdb


class model(nn.Module):
    def __init__(self, opt):
        super(model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.Generator(opt)
        if opt.phase == "train":
            self.netD = discriminators.Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)

    def forward(self, real, rendered, mask, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(rendered)
            output_D = self.netD(fake)
            #loss_G_adv = losses_computer.loss(output_D, mask, for_real=True)
            #loss_G_adv = losses_computer.loss(output_D, real, for_real=True)
            pdb.set_trace()
            loss_G_adv = losses_computer.loss(output_D, real, mask, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, real)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(rendered)
            output_D_fake = self.netD(fake)
            #loss_D_fake = losses_computer.loss(output_D_fake, mask, for_real=False)
            #loss_D_fake = losses_computer.loss(output_D_fake, real, for_real=False)
            loss_D_fake = losses_computer.loss(output_D_fake, real, mask, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(real)
            #loss_D_real = losses_computer.loss(output_D_real, real, for_real=True)
            loss_D_real = losses_computer.loss(output_D_real, real, mask, for_real=True)
            loss_D += loss_D_real
            return loss_D, [loss_D_fake, loss_D_real]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(rendered)
                else:
                    fake = self.netEMA(rendered)
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['mask'] = data['mask'].long()
    if opt.gpu_ids != "-1":
        data['mask'] = data['mask'].cuda()
    mask_map = data['mask']
    bs, _, h, w = mask_map.size()
    if opt.gpu_ids != "-1":
        mask = torch.cuda.FloatTensor(bs, 1, h, w).zero_()
    else:
        mask = torch.FloatTensor(bs, 1, h, w).zero_()
    new_mask = mask.scatter_(1, mask_map, 1.0)
    return new_mask

