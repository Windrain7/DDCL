import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Downsample, LayerNorm, Upsample


class Gen(nn.Module):
    def __init__(
        self,
        inp_channels=6,
        out_channels=3,
        dim=48,
        num_blocks=[1, 2, 2],
        num_latent_blocks=4,
        num_refine_blocks=2,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super().__init__()
        self.embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=bias)
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, num_block in enumerate(num_blocks):
            encoder = nn.Sequential(*[ForwardBlock(dim * (2**i), ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_block)])
            down = Downsample(dim * (2**i))
            self.encoder.append(encoder)
            self.down.append(down)

        self.latent = nn.Sequential(
            *[
                ForwardBlock(
                    dim * (2 ** len(num_blocks)),
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                )
                for _ in range(num_latent_blocks)
            ]
        )

        self.up = nn.ModuleList()
        self.reduce_chan = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i, num_block in enumerate(num_blocks[::-1]):
            k = len(num_blocks) - i
            up = Upsample(dim * (2**k))
            reduce_chan = nn.Conv2d(dim * (2**k), dim * (2 ** (k - 1)), kernel_size=1, bias=bias)
            decoder = nn.Sequential(*[ForwardBlock(dim * (2 ** (k - 1)), ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_block)])
            self.up.append(up)
            self.reduce_chan.append(reduce_chan)
            self.decoder.append(decoder)
        self.refine = nn.Sequential(*[ForwardBlock(dim, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_refine_blocks)])
        self.out = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward_encode(self, inp: torch.Tensor):
        skips = []
        for encoder, down in zip(self.encoder, self.down):
            inp = encoder(inp)
            skips.append(inp)
            inp = down(inp)
        return inp, skips

    def forward_decode(self, inp: torch.Tensor, skips: list):
        for up, reduce_chan, decoder, skip in zip(self.up, self.reduce_chan, self.decoder, skips[::-1]):
            inp = up(inp)
            inp = torch.cat([inp, skip], dim=1)
            inp = reduce_chan(inp)
            inp = decoder(inp)
        return inp

    def forward(self, inp: torch.Tensor, mask: torch.Tensor):
        x = inp
        inp = self.embed(torch.cat([inp, mask], dim=1))
        inp, skips = self.forward_encode(inp)
        inp = self.latent(inp)
        inp = self.forward_decode(inp, skips)
        inp = self.refine(inp)
        inp = self.out(inp)
        out = x + inp
        return out


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, inp):
        x = self.project_in(inp)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ForwardBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.ffn = FFN(dim, ffn_expansion_factor, bias)
        self.norm = LayerNorm(dim, LayerNorm_type)

    def forward(self, inp):
        inp = self.norm(inp)  # 要是不过norm层直接加inp，模型就不学习了
        x = self.ffn(inp)
        return x + inp
