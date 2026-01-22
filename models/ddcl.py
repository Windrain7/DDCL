import pickle

import torch
import torch.nn as nn

from .archs.discriminator import *
from .archs.generator import Gen
from .archs.moco import *
from .archs.utils import *
from .archs.vgg16 import Vgg16
from .losses.loss import *


class DDCL(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.is_test = opts.is_test
        # self.gpu = opts.gpu

        # discriminators
        self.disA = MultiScaleDis(3, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm, FFT_mode=opts.dis_FFT_mode)
        self.disB = MultiScaleDis(3, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm, FFT_mode=opts.dis_FFT_mode)

        # rse
        self.rse = MoCo(
            base_encoder=ResEncoder(num_layers=opts.moco_num_layers, FFT_mode=opts.moco_FFT_mode),
            base_decoder=ResDecoder(num_layers=opts.moco_num_layers[::-1]),
            dim=opts.moco_dim,
            K=opts.moco_K,
        )

        # generator
        self.genA = Gen()
        self.genB = Gen()

        # vgg
        self.vgg = Vgg16()
        self.vgg.load()

        if self.is_test:
            return

        # optimizers
        lr = opts.lr
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.rse_opt = torch.optim.Adam(self.rse.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.genA_opt = torch.optim.Adam(self.genA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.genB_opt = torch.optim.Adam(self.genB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(opts.gan_mode).cuda(opts.gpu)
        self.criterionFreqMag = FreqMagLoss(opts.low_th, opts.crs_th, opts.std_th, opts.window_size, opts.extract_out)
        self.criterionCE = nn.CrossEntropyLoss()

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(opts.pool_size, 3, opts.patch_size).cuda(opts.gpu)
        self.fake_B_pool = ImagePool(opts.pool_size, 3, opts.patch_size).cuda(opts.gpu)

    def initialize(self):
        self.disA.apply(gaussian_weights_init)
        self.disB.apply(gaussian_weights_init)
        self.rse.apply(gaussian_weights_init)
        self.genA.apply(gaussian_weights_init)
        self.genB.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = get_scheduler(self.disB_opt, opts, last_ep)
        self.rse_sch = get_scheduler(self.rse_opt, opts, last_ep)
        self.genA_sch = get_scheduler(self.genA_opt, opts, last_ep)
        self.genB_sch = get_scheduler(self.genB_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.rse.cuda(self.gpu)
        self.genA.cuda(self.gpu)
        self.genB.cuda(self.gpu)
        self.vgg.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def feed_data(self, input_A, input_B):
        if type(input_A) in (list, tuple):
            self.input_A = [x.cuda(self.gpu) for x in input_A]
            self.input_B = [x.cuda(self.gpu) for x in input_B]
        else:
            self.input_A = input_A.cuda(self.gpu)
            self.input_B = input_B.cuda(self.gpu)

    def test_forward(self, a2b=True):
        if a2b:
            self.mask_a_0 = self.rse.test_forward(self.input_A)
            out = self.genA.forward(self.input_A, self.mask_a_0)
        else:
            self.mask_a_0 = self.rse.test_forward(self.input_A)
            noise = torch.randn_like(self.mask_a_0, device=self.mask_a_0.device) * 0.01
            self.mask_b_0 = self.mask_a_0 + noise
            out = self.genB.forward(self.input_B, self.mask_b_0)
        return out

    def train_forward(self, ep, opts):
        """self.real_A_encoded -> self.fake_A_encoded -> self.real_A_recon"""
        """self.real_B_encoded -> self.fake_B_encoded -> self.real_B_recon"""
        self.real_A_encoded = self.input_A[0]
        self.real_B_encoded = self.input_B[0]

        # get first cycle
        """self.real_A_encoded -> self.fake_A_encoded"""
        """self.real_B_encoded -> self.fake_B_encoded"""
        self.mask_a_0, self.logits_a, self.labels_a = self.rse.train_forward(self.real_A_encoded, self.input_A[1])
        self.fake_A_encoded = self.genA.forward(self.real_A_encoded, self.mask_a_0)
        self.mask_b_0 = self.rse.test_forward(self.real_B_encoded)
        self.fake_B_encoded = self.genB.forward(self.real_B_encoded, self.mask_a_0)

        # get perceptual loss
        self.perc_real_A = self.vgg(self.real_A_encoded).detach()
        self.perc_fake_A = self.vgg(self.fake_A_encoded).detach()

        # get second cycle
        """self.fake_A_encoded -> self.real_A_recon"""
        """self.fake_B_encoded -> self.real_B_recon"""
        # The mask for fake_B_encoded is theoretically mask_a. They are not negative samples of each other.
        # since mask_a has already been queued, test_forward is used.
        self.mask_b_1 = self.rse.test_forward(self.fake_B_encoded)
        self.real_B_recon = self.genA.forward(self.fake_B_encoded, self.mask_b_1)
        self.mask_a_1 = self.rse.test_forward(self.fake_A_encoded)
        self.real_A_recon = self.genB.forward(self.fake_A_encoded, self.mask_b_1)

    def update_D(self, opts):
        fake_A_encoded = self.fake_A_pool(self.fake_A_encoded)
        fake_B_encoded = self.fake_B_pool(self.fake_B_encoded)

        # update disA
        self.disA_opt.zero_grad()
        self.loss_D1_A = self.backward_D_basic(self.disA, self.real_A_encoded, fake_B_encoded)
        self.disA_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        self.loss_D1_B = self.backward_D_basic(self.disB, self.real_B_encoded, fake_A_encoded)
        self.disB_opt.step()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real1 = self.criterionGAN(pred_real[0], True)
        loss_D_real2 = self.criterionGAN(pred_real[1], True)
        loss_D_real3 = self.criterionGAN(pred_real[2], True)
        loss_D_real = (loss_D_real1 + loss_D_real2 + loss_D_real3) / 3

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake[0], False)
        loss_D_fake2 = self.criterionGAN(pred_fake[1], False)
        loss_D_fake3 = self.criterionGAN(pred_fake[2], False)
        loss_D_fake = (loss_D_fake1 + loss_D_fake2 + loss_D_fake3) / 3

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def update_EG(self, ep, opts):
        self.train_forward(ep, opts)
        self.rse_opt.zero_grad()
        self.genA_opt.zero_grad()
        self.genB_opt.zero_grad()
        self.backward_EG(ep, opts)
        self.rse_opt.step()
        self.genA_opt.step()
        self.genB_opt.step()

    def backward_EG(self, ep, opts):
        self.loss_G = 0
        # adversarial loss
        self.loss_G_GAN_A = self.criterionGAN(self.disA(self.fake_B_encoded)[0], True) * opts.GAN_lambda
        self.loss_G_GAN_B = self.criterionGAN(self.disB(self.fake_A_encoded)[0], True) * opts.GAN_lambda
        self.loss_G += self.loss_G_GAN_A + self.loss_G_GAN_B

        # cross cycle consistency loss
        self.loss_G_L1_A = self.criterionL1(self.real_A_recon, self.real_A_encoded) * opts.cycle_lambda
        self.loss_G_L1_B = self.criterionL1(self.real_B_recon, self.real_B_encoded) * opts.cycle_lambda
        self.loss_G += self.loss_G_L1_A + self.loss_G_L1_B

        # perceptual loss
        self.loss_perceptual = self.criterionL2(self.perc_fake_A, self.perc_real_A) * opts.perceptual_lambda
        self.loss_G += self.loss_perceptual

        # mask loss
        fake = self.fake_A_encoded + self.mask_a_0
        self.loss_mask_a_0 = self.criterionL2(fake, self.real_A_encoded) * opts.mask_lambda

        self.loss_mask_b_0 = self.criterionL2(self.mask_b_0, torch.zeros_like(self.mask_b_0, device=self.mask_b_0.device)) * opts.mask_lambda
        self.loss_G += self.loss_mask_a_0 + self.loss_mask_b_0

        if opts.sec_mask_loss:
            self.loss_mask_a_1 = self.criterionL2(self.mask_a_1, torch.zeros_like(self.mask_a_1, device=self.mask_a_1.device)) * opts.mask_lambda
            self.loss_mask_b_1 = self.criterionL2(self.real_B_recon + self.mask_b_1, self.fake_B_encoded) * opts.mask_lambda

            self.loss_G += self.loss_mask_a_1 + self.loss_mask_b_1

        # frequency magnitude loss
        if self.criterionFreqMag:
            self.loss_freq_mag = self.criterionFreqMag(self.fake_A_encoded, self.mask_a_0.detach()) * opts.freq_lambda
            self.loss_G += self.loss_freq_mag

        # contrast loss
        self.loss_contrast = self.criterionCE(self.logits_a, self.labels_a) * opts.contrast_lambda
        self.loss_G += self.loss_contrast

        self.loss_G.backward()

    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.rse_sch.step()
        self.genA_sch.step()
        self.genB_sch.step()

    def resume(self, model_dir, train=True):
        # weight and pool
        checkpoint = torch.load(model_dir)
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.criterionGAN.load_state_dict(checkpoint['criterionGAN'])
            self.fake_A_pool.load_state_dict(checkpoint['fake_A_pool'])
            self.fake_B_pool.load_state_dict(checkpoint['fake_B_pool'])
        self.rse.load_state_dict(checkpoint['atten'])
        self.genA.load_state_dict(checkpoint['genA'])
        self.genB.load_state_dict(checkpoint['genB'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.rse_opt.load_state_dict(checkpoint['atten_opt'])
            self.genA_opt.load_state_dict(checkpoint['genA_opt'])
            self.genB_opt.load_state_dict(checkpoint['genB_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'disB': self.disB.state_dict(),
            'criterionGAN': self.criterionGAN.state_dict(),
            'fake_A_pool': self.fake_A_pool.state_dict(),
            'fake_B_pool': self.fake_B_pool.state_dict(),
            'atten': self.rse.state_dict(),
            'genA': self.genA.state_dict(),
            'genB': self.genB.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'atten_opt': self.rse_opt.state_dict(),
            'genA_opt': self.genA_opt.state_dict(),
            'genB_opt': self.genB_opt.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, filename)
        return

    def save_dict(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dict(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def assemble_outputs(self):
        images_a_1 = self.normalize_image(self.input_A[1]).detach()
        images_a_0 = self.normalize_image(self.real_A_encoded).detach()
        images_a_01 = self.normalize_image(self.fake_A_encoded).detach()
        images_a_02 = self.normalize_image(self.real_A_recon).detach()
        images_mask_a = self.normalize_image(self.mask_a_0).detach()
        images_b_1 = self.normalize_image(self.input_B[1]).detach()
        images_b_0 = self.normalize_image(self.real_B_encoded).detach()
        images_b_01 = self.normalize_image(self.fake_B_encoded).detach()
        images_b_02 = self.normalize_image(self.real_B_recon).detach()
        images_mask_b = self.normalize_image(self.mask_b_0).detach()

        row1 = torch.cat((images_a_1[0], images_a_0[0], images_a_01[0], images_a_02[0], images_mask_a[0]), 2)
        row2 = torch.cat((images_b_1[0], images_b_0[0], images_b_01[0], images_b_02[0], images_mask_b[0]), 2)
        return torch.cat((row1, row2), 1)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]
