import torch.nn.init as init
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


# Initialization for Convolutional Sensing Module
def _csm_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=1/math.sqrt(m.weight.shape[-1]*m.weight.shape[-2]*m.weight.shape[-3]))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class RKB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True):
        super(RKB, self).__init__()

        self.k1 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias), nn.PReLU(n_feats, 0.25))
        self.k2 = nn.Sequential(conv(n_feats, n_feats, kernel_size, bias=bias), nn.PReLU(n_feats, 0.25))
        self.alpha = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def forward(self, x):
        b_1 = torch.exp(-torch.exp(self.alpha))
        b_2 = 1-b_1
        a_21 = 1 / (2*b_2)
        
        x_k1 = self.k1(self.k1(x))
        x_k1 = a_21 * x_k1 + x

        x_k2 = self.k2(self.k2(x_k1))
        x_k2 = b_2 * x_k2 + b_1 * x_k1
        
        out = x_k2 + x
        return out


class RKCCSNet(nn.Module):
    def __init__(self, sensing_rate, conv=default_conv):
        super(RKCCSNet, self).__init__()
        self.measurement = int(sensing_rate * 1024)
        
        n_resblocks = 8
        n_feats_csm = 64
        n_feats = 64
        kernel_size = 3

        # define head module
        m_head = [conv(1, n_feats, kernel_size)]

        # define body module
        m_body = [RKB( conv, n_feats, kernel_size) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, 1, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if sensing_rate == 0.5:
            # 0.5000
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(2, 4, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 2
        elif sensing_rate == 0.25:
            # 0.2500
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(4, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
        elif sensing_rate == 0.125:
            # 0.1250
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(2, 16, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 4
        elif sensing_rate == 0.0625:
            # 0.0625
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(4, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
        elif sensing_rate == 0.03125:
            # 0.03125
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 2, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 8
        elif sensing_rate == 0.015625:
            # 0.015625
            self.csm = nn.Sequential(nn.Conv2d(1, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, n_feats_csm, kernel_size=3, padding=1, stride=2, bias=False),
                                        nn.Conv2d(n_feats_csm, 4, kernel_size=1, padding=0, stride=1, bias=False))
            self.initial = nn.Conv2d(4, 256, kernel_size=1, padding=0, stride=1, bias=False)
            self.m = 16

        self.csm.apply(_csm_init)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.unfold(x, kernel_size=32, stride=32)
        x = x.permute(0, 2, 1).reshape(-1, 1, 32, 32)
        x = self.csm(x)
        x = self.initial(x)
        x = nn.PixelShuffle(self.m)(x)
        x = x.reshape(n, -1, 1024).permute(0, 2, 1)
        initial = F.fold(x, (h, w), kernel_size=32, stride=32)

        x = self.head(initial)
        res = self.body(x)
        res += x

        x = self.tail(res) + initial

        return x, initial
