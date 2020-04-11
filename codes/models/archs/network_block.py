import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class ResidualBlock_noBN_CA(nn.Module):
    ''' Residual block w/o BN + channel attention
    ----Conv--ReLU--Conv--------------------------------------x---concat--conv-----
      |               |----Avg pool--FC--ReLU--FC--sigmoid----|----|           |
      |------------------------------------------------------------------------|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN_CA, self).__init__()
        reduction_ratio = 16
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(2 * nf, nf, 1, 1, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(nf, nf // reduction_ratio, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(nf // reduction_ratio, nf, bias=False),
            nn.Sigmoid()
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization: residual learning
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        arch_util.initialize_weights([self.fc], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        identity_out = out
        B, C, _, _ = out.size()
        att = self.fc(self.avg_pool(out).view(B, C)).view(B, C, 1, 1)
        out = out * att.expand_as(out)
        out = self.conv3(torch.cat([identity_out, out], dim=1))

        out += identity

        return out



