import functools

import torch
import torch.nn as nn

from .utils import gaussian_weights_init, spectral_norm


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False, FFT_mode=0):
        super().__init__()
        ch = 64
        self.FFT_mode = FFT_mode
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim * 2 if FFT_mode else input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, inp):
        outs = []
        for Dis in self.Diss:
            if self.FFT_mode == 1:
                x = torch.cat([inp, torch.log(torch.fft.fft2(inp).abs() + 1)], dim=1)
            elif self.FFT_mode == 2:
                x = torch.cat([inp, torch.fft.fft2(inp).abs()], dim=1)
            else:
                x = inp
            outs.append(Dis(x))
            inp = self.downsample(inp)
        return outs


MultiScaleDisWithFFT = functools.partial(MultiScaleDis, FFT_mode=1)
MultiScaleDisWithFFTV2 = functools.partial(MultiScaleDis, FFT_mode=2)


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super().__init__()
        model = []
        model.append(nn.ReflectionPad2d(padding))
        conv = nn.Conv2d(n_in, n_out, kernel_size, stride, padding=0, bias=True)
        if sn:
            model.append(spectral_norm(conv))
        else:
            model.append(conv)
        if 'norm' == 'Instance':
            model.append(nn.InstanceNorm2d(n_out, affine=False))
        model.append(nn.LeakyReLU(inplace=True))
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)
