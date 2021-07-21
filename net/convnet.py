import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        bn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_block2(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        bn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        bn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def conv_block3(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        bn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        bn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1))
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, cla_num=4):
        super().__init__()
        dims = [32, 64, 128, 256, 512]
        self.encoder = nn.Sequential(
            conv_block(x_dim, dims[0]),
            conv_block2(dims[0], dims[1]),
            conv_block2(dims[1], dims[2]),
            conv_block2(dims[2], dims[3]),
            conv_block3(dims[3], dims[4]),
        )
        self.linear = nn.Linear(dims[4], cla_num)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(x.size(0), -1)

