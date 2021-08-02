import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader
from utils import train

def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1), nn.ReLU(inplace=True)]
    # inplace 的作用是可以直接原地覆蓋一下上一層的數據

    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        net.append(nn.ReLU(True))

    net.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

    return nn.Sequential(*net)

def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


class VggNet16(nn.Module):
    def __init__(self):
        super(VggNet16, self).__init__()
        self.conv1 = vgg_stack((1, 1, 2, 3, 3), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
        # 第一个括号内表示的是总共有5个block，每个block中含有两到三个卷积层， Vgg的卷积操作是每次经过一个block就会除2,
        # 第二个括号内就表示5个block中输入的channel和输出的channel的值。
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def data_tf(x):
    x = np.array(x, dtype='float32') / 255  # 图片颜色的最大值为255，这个可以让数据缩小到0-1
    x = (x - 0.5) / 0.5  # 标准化处理， 让在-1到1之间
    x = x.transpose((2, 0, 1))  # 转换维度，pytorch需要让通道数在第一个维度出现
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data', train=True, transform=data_tf, download=False)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10('./data', train=False, transform=data_tf, download=False)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
net = VggNet16()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()
train(net, train_data, test_data, num_epochs=30, optimizer=optimizer, criterion=criterion)
