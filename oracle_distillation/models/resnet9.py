import torch.nn as nn


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, pools=0):
    layers = [
        nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    ]
    if pools > 0:
        layers.append(nn.MaxPool2d(pools))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = conv_bn(3, 64)
        self.layer1 = conv_bn(64, 128, pools=2)
        self.res1 = Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128)))
        self.layer2 = conv_bn(128, 256, pools=2)
        self.layer3 = conv_bn(256, 512, pools=2)
        self.res3 = Residual(nn.Sequential(conv_bn(512, 512), conv_bn(512, 512)))
        self.pool = nn.MaxPool2d(4)
        self.classifier = nn.Sequential(Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res3(out)
        out = self.pool(out)
        out = self.classifier(out)
        return out
