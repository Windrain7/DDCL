# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Downsample, LayerNorm, Upsample


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, base_decoder, dim=256, K=3072, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 256)
        K: queue size; number of negative keys (default: 3072)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.decoder = base_decoder

        # create the queue
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def train_forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        fq, skips, q = self.encoder_q.train_forward(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, _, k = self.encoder_k.train_forward(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        out = self.decoder(fq, skips)

        return out, logits, labels

    def test_forward(self, x):
        f, skips = self.encoder_q.test_forward(x)
        out = self.decoder(f, skips)
        return out


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = nn.Sequential(
            LayerNorm(dim, 'WithBias'),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            LayerNorm(dim, 'WithBias'),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            LayerNorm(dim, 'WithBias'),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return F.gelu(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self, inp_dim=3, dim=32, num_layers=[1, 1, 1], latent_num=1, FFT_mode=0) -> None:
        super().__init__()
        self.FFT_mode = FFT_mode
        self.embed = nn.Conv2d(inp_dim * 2 if FFT_mode in (1, 2) else inp_dim, dim, kernel_size=3, padding=1, bias=False)
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, nums in enumerate(num_layers):
            self.layers.append(nn.Sequential(*[ResBlock(dim * (2**i)) for _ in range(nums)]))
            self.downs.append(Downsample(dim * (2**i)))
        self.mids = nn.Sequential(*[ResBlock(dim * (2**len(num_layers))) for _ in range(latent_num)])
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(dim * 2**len(num_layers), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def train_forward(self, x):
        x, skips = self.test_forward(x)
        return x, skips, self.mlp(x)

    def test_forward(self, x):
        if self.FFT_mode == 1:
            x = torch.cat([x, torch.log(torch.fft.fft2(x).abs() + 1)], dim=1)
        elif self.FFT_mode == 2:
            x = torch.cat([x, torch.fft.fft2(x).abs()], dim=1)
        elif self.FFT_mode == 3:
            x = torch.log(torch.fft.fft2(x).abs() + 1)
        x = self.embed(x)
        skips = []
        for layer, down in zip(self.layers, self.downs):
            x = layer(x)
            skips.append(x)
            x = down(x)
        x = self.mids(x)
        return x, skips


class ResDecoder(nn.Module):
    def __init__(self, out_dim=3, dim=32, num_layers=[1, 1, 1]) -> None:
        super().__init__()
        self.ups = nn.ModuleList()
        self.reduce_chan = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i, nums in enumerate(num_layers):
            k = len(num_layers) - i
            self.ups.append(Upsample(dim * (2**k)))
            self.reduce_chan.append(nn.Conv2d(dim * (2**k), dim * (2 ** (k - 1)), kernel_size=1, bias=False))
            self.layers.append(nn.Sequential(*[ResBlock(dim * (2 ** (k - 1))) for _ in range(nums)]))
        self.out = nn.Conv2d(dim, out_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x, skips):
        for up, reduce_chan, layer, skip in zip(self.ups, self.reduce_chan, self.layers, skips[::-1]):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = reduce_chan(x)
            x = layer(x)
        return self.out(x)


