"""FireCast network: a small U-Net on 90 m patches with scalar conditioning (weather + fire scalars).
The full-resolution path is thin (stem + refine); most work happens at 180-1440 m. Scalars enter through an MLP
that adds a per-channel bias at every level. BatchNorm is folded into the convs for the numpy release."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class FireCastNet(nn.Module):
    def __init__(self, in_ch, n_s, ch=(16, 24, 32, 48, 48), hid=64):
        super().__init__()
        c0, c1, c2, c3, c4 = ch
        self.ch = ch
        self.stem = cbr(in_ch, c0)
        self.e1 = nn.Sequential(cbr(c0, c1, 2), cbr(c1, c1))
        self.e2 = nn.Sequential(nn.MaxPool2d(2), cbr(c1, c2), cbr(c2, c2))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), cbr(c2, c3), cbr(c3, c3))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), cbr(c3, c4), cbr(c4, c4))
        self.d3 = cbr(c4 + c3, c3)
        self.d2 = cbr(c3 + c2, c2)
        self.d1 = cbr(c2 + c1, c1)
        self.d0 = cbr(c1 + c0, c0)
        self.head = nn.Conv2d(c0, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(n_s, hid), nn.ReLU(inplace=True), nn.Linear(hid, hid), nn.ReLU(inplace=True))
        self.film = nn.ModuleList([nn.Linear(hid, c) for c in (c0, c1, c2, c3, c4, c3, c2, c1, c0)])
        self.head_s = nn.Linear(hid, 1)

    def forward(self, x, s):
        h = self.mlp(s)
        b = [f(h)[:, :, None, None] for f in self.film]
        x0 = self.stem(x) + b[0]
        x1 = self.e1(x0) + b[1]
        x2 = self.e2(x1) + b[2]
        x3 = self.e3(x2) + b[3]
        x4 = self.e4(x3) + b[4]
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        y = self.d3(torch.cat([up(x4), x3], 1)) + b[5]
        y = self.d2(torch.cat([up(y), x2], 1)) + b[6]
        y = self.d1(torch.cat([up(y), x1], 1)) + b[7]
        y = self.d0(torch.cat([up(y), x0], 1)) + b[8]
        return self.head(y)[:, 0] + self.head_s(h)[:, :, None]


def n_params(m):
    return sum(p.numel() for p in m.parameters())
