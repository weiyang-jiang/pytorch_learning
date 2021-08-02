import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader
from utils import *

# DenseNet 可以保证底层的输出会保留进入到所有后面的层，可以更好的保证梯度的传播， 同时可以让低层的特征和高层的一起训练， 能够得到更好的效果
# 定义了一个含有BN和激活层的卷积模块, 经过此卷积SIZE不变
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, bias=False)
    )
    return layer

# 定义了一个dense连接模块,这是一种多层跳连的结构
class denseblock(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(denseblock, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate

        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

# 为了防止dense所导出的模块的channel值过大,引入一个转换卷积降低通道数.
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, (1, 1)),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer


class denseNet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(denseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )

        channels = 64
        block = []
        # 这其中总共有四个dense, 在每个dense结束的时候都会加上一个降低通道数的模块.(除了最后一个dense)
        for i, layers in enumerate(block_layers):
            block.append(denseblock(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition(channels, channels // 2))
                channels = channels // 2

        self.block2 = nn.Sequential(*block)
        self.block2.add_module("bn", nn.BatchNorm2d(channels))
        self.block2.add_module("relu", nn.ReLU(True))
        self.block2.add_module("avg_pool", nn.AvgPool2d(3))
        self.FC = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x



net = denseNet(3, 10)
# test_x = Variable(torch.zeros(1, 3, 96, 96))
# test_y = test_net(test_x)
summary(net.cuda(), (3, 96, 96))
train_set = CIFAR10("./data", train=True, transform=data_tf, download=False)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10("./data", train=False, transform=data_tf, download=False)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train(net, train_data, test_data, 30, optimizer, criterion)
