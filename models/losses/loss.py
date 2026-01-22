import random

import torch
import torch.nn.functional as F
from torch import nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # pdb.set_trace()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ImagePool(nn.Module):
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, input_dim, patch_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer
        """
        super().__init__()
        self.register_buffer('num_images', torch.tensor(0, dtype=torch.int))
        self.register_buffer('images', torch.Tensor(pool_size, input_dim, patch_size, patch_size))

    def forward(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.images.shape[0] == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = image.detach()
            if self.num_images < self.images.shape[0]:  # if the buffer is not full; keep inserting current images to the buffer
                self.images[self.num_images] = image
                self.num_images += 1
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.images.shape[0] - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.stack(return_images, 0)  # collect all the images and return
        return return_images


class FreqMagLoss(nn.Module):
    def __init__(self, low_th=0.1, crs_th=0.03, std_th=1.5, window_size=53, extract_out=True):
        super().__init__()
        self.low_th = low_th
        self.std_th = std_th
        self.crs_th = crs_th
        self.window_size = window_size
        self.extract_out = extract_out

    def get_filter_mask(self, inp):
        H, W = inp.shape[-2:]
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.to(inp.device), x.to(inp.device)
        distance = torch.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        radius = min(H, W) * self.low_th
        filter_mask = distance > radius
        h, w = int(H * self.crs_th), int(W * self.crs_th)
        filter_mask[..., H // 2 - h : H // 2 + h, :] = 0
        filter_mask[..., :, W // 2 - w : W // 2 + w] = 0
        return filter_mask

    def preprocess(self, inp: torch.Tensor):
        inp_fft_mag = torch.fft.fft2(inp).abs()
        inp_fft_mag = torch.fft.fftshift(inp_fft_mag, dim=(-2, -1))
        inp_fft_mag = torch.log(inp_fft_mag + 1)
        return inp_fft_mag

    def extract(self, inp: torch.Tensor):
        mean = F.avg_pool2d(inp, self.window_size, stride=1, padding=self.window_size // 2)
        mean_of_square = F.avg_pool2d(inp**2, self.window_size, stride=1, padding=self.window_size // 2)
        std = torch.sqrt(torch.clamp(mean_of_square - mean**2, min=1e-10))
        mask = inp > mean + self.std_th * std
        return inp * mask * self.get_filter_mask(inp)

    def forward(self, out: torch.Tensor, mask: torch.Tensor):
        mask_fft_mag = self.preprocess(mask)
        mask_fft_mag = self.extract(mask_fft_mag)

        out_fft_mag = self.preprocess(out)
        if self.extract_out:
            out_fft_mag = self.extract(out_fft_mag)

        loss = torch.mean(mask_fft_mag * out_fft_mag)
        return loss


class FFTLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp, tar):
        x_fft = torch.fft.rfft2(inp, dim=(-2, -1))
        target_fft = torch.fft.rfft2(tar, dim=(-2, -1))
        return F.l1_loss(x_fft.real, target_fft.real) + F.l1_loss(x_fft.imag, target_fft.imag)
