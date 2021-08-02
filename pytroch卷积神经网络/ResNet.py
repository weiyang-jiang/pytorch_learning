import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader
from utils import *

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.a1 = nn.ReLU(True)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.a2 = nn.ReLU(True)

        if not self.same_shape:  # 如果要改变size的大小，就要让stride=2,然后就可以让图像大小缩小一倍。
            self.conv3 = nn.Conv2d(in_channel, out_channel, (1, 1), stride=stride)

        self.a3 = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.a1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.a2(out)
        if not self.same_shape:
            x = self.conv3(x)
        x = self.a3(out + x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(ResNet, self).__init__()
        self.block1 = nn.Conv2d(in_channel, 64, (7, 7), stride=2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )

        self.classifer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer(x)
        return x




net = ResNet(3, 10)
# test_x = Variable(torch.zeros(1, 3, 96, 96))
# test_y = test_net(test_x)
summary(net.cuda(), (3, 96, 96))
train_set = CIFAR10("./data", train=True, transform=train_tf, download=False)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10("./data", train=False, transform=test_tf, download=False)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train(net, train_data, test_data, 30, optimizer, criterion)