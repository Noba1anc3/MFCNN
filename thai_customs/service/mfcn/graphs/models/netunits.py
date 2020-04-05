
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger


class DoubleConv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch, d):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=d, padding=d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=d, padding=d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, d):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, d)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, d):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, d)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class DoubleConv2(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsamp(nn.Module):
    def __init__(self, unpool=False):
        super(Upsamp, self).__init__()

        if unpool:
            self.up = nn.MaxUnpool2d(2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return x


class Resize(nn.Module):
    def __init__(self):
        super(Resize, self).__init__()

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        dim = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]

        x1 = F.pad(x1, dim)

        return x1



class Up(nn.Module):
    def __init__(self, in_ch, out_ch, unpool=False, bilinear=True):
        super(Up, self).__init__()


        # if unpool:
        #     self.up = nn.MaxUnpool2d(2)
        # elif bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv2(in_ch, out_ch)


    def forward(self, x1, x2):
        # x1 = self.up(x1)

        # front map is larger?????
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # dim = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        #
        # x1 = F.pad(x1, dim)

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
