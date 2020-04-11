import functools
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import models.archs.network_block as network_block


class Coarse_Fine_RCANNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, upscale=4):
        super(Coarse_Fine_RCANNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_third = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(network_block.ResidualBlock_noBN_CA, nf=nf)

        self.recon_trunk1 = arch_util.make_layer(basic_block, 5)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 5)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 6)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 8:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv3 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.conv_second, self.conv_third, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)
        if self.upscale == 8:
            arch_util.initialize_weights(self.upconv3, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out1 = self.recon_trunk1(fea)
        out1 = self.lrelu(self.pixel_shuffle(self.upconv1(out1)))
        out2 = self.lrelu(self.conv_second(out1))
        out2 = self.recon_trunk2(out2)
        out2 = self.lrelu(self.pixel_shuffle(self.upconv2(out2)))
        out3 = self.lrelu(self.conv_third(out2))
        out3 = self.recon_trunk3(out3)
        out3 = self.lrelu(self.pixel_shuffle(self.upconv3(out3)))

        out = self.conv_last(self.lrelu(self.HRconv(out3)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
