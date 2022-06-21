from models.SeqAttention import SeqAttention
import torch
import torch.nn as nn
import math


def layerInitializer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, padding=1, mpool=True):
        super(ConvBlock, self).__init__()
        if mpool:
            self.blocks = [nn.Conv2d(in_dim, hid_dim, kernel_size=3, padding=padding),
                           nn.BatchNorm2d(hid_dim),
                           nn.ReLU(),
                           nn.MaxPool2d(2)]
        else:
            self.blocks = [nn.Conv2d(in_dim, hid_dim, kernel_size=3, padding=1),
                           nn.BatchNorm2d(hid_dim),
                           nn.ReLU()]

        for layer in self.blocks:
            layerInitializer(layer)

        self.convBlock = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.convBlock(x)


class SetFeat4(nn.Module):
    def __init__(self, n_filters, n_heads, enc_out_chanal, sqa_type):
        super(SetFeat4, self).__init__()
        self.layer1 = ConvBlock(3, n_filters[0])
        self.layer2 = ConvBlock(n_filters[0], n_filters[1])
        self.layer3 = ConvBlock(n_filters[1], n_filters[2])
        self.layer4 = ConvBlock(n_filters[2], n_filters[3], mpool=True)
        residual_mode = False
        self.atten1 = SeqAttention(n_filters[0], enc_out_chanal, n_heads[0], sqa_type, residual_mode)
        self.atten2 = SeqAttention(n_filters[1], enc_out_chanal, n_heads[1], sqa_type, residual_mode)
        self.atten3 = SeqAttention(n_filters[2], enc_out_chanal, n_heads[2], sqa_type, residual_mode)
        self.atten4 = SeqAttention(n_filters[3], enc_out_chanal, n_heads[3], sqa_type, residual_mode)

    def forward(self, x):
        x = self.layer1(x)
        a1 = self.atten1(x)
        x = self.layer2(x)
        a2 = self.atten2(x)
        x = self.layer3(x)
        a3 = self.atten3(x)
        x = self.layer4(x)
        a4 = self.atten4(x)
        return torch.cat((a1, a2, a3, a4), dim=1)




