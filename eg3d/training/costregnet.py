import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN

from ipdb import set_trace as st

#----------------------------------------
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.act = nn.LeakyReLU()
        # self.bn = nn.ReLU()
        # self.conv.apply(conv3d_weights_init)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CostRegNet_Deeper(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet_Deeper, self).__init__()
        out_dim=8
        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        # if self.conv3.bn.weight.grad != None:
        # st()

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))
        # print("CostRegNetDeeper bottleneck:", conv62.shape): # 256^3 -> 8^3; 128^3 -> 4^3
        x = conv61 + self.conv27(conv62)
        x = conv6 + self.conv17(x)

        x = conv4 + self.conv7(x)
        # del conv4
        x = conv2 + self.conv9(x)
        # x = conv2 + self.conv9(conv4)
        del conv2, conv4
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x