import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader
from utils import train

#　定义一个卷积加一个BN层和激活层
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    """
    输入：输入数据x1…xm（这些数据是准备进入激活函数的数据）
        计算过程中可以看到,
        1.求数据均值
        2.求数据方差
        3.数据进行标准化（个人认为称作正态化也可以）
        4.训练参数γ，β
        5.输出y通过γ与β的线性变换得到新的值

    在深度网络中，如果网络的激活输出很大，其梯度就很小，学习速率就很慢。假设每层学习梯度都小于最大值0.25，网络有n层，因为链式求导的原因，第一层的梯度小于0.25的n次方，所以学习速率就慢，对于最后一层只需对自身求导1次，梯度就大，学习速率就快。
这会造成的影响是在一个很大的深度网络中，浅层基本不学习，权值变化小，后面几层一直在学习，结果就是，后面几层基本可以表示整个网络，失去了深度的意义。

关于梯度爆炸，根据链式求导法，
第一层偏移量的梯度=激活层斜率1x权值1x激活层斜率2x…激活层斜率(n-1)x权值(n-1)x激活层斜率n
假如激活层斜率均为最大值0.25，所有层的权值为100，这样梯度就会指数增加。
    """
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(inplace=True)
    )
    return layer
# Inception(192, 64, 96, 128, 16, 32, 32),
class Inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(Inception, self).__init__()
        # 第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, (1, 1))
        
        # 第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, (1, 1)),
            conv_relu(out2_1, out2_3, (3, 3), padding=1)
        )
        
        # 第三条线路
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, (1, 1)),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )
        
        # 第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, (1, 1))
        )
    
    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output
    
class GoogleNet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(GoogleNet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64, kernel=(7, 7), stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        
        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            # 192x11x11 -> 64x11x11
            #                 +
# 192x11x11 -> 96x11x11 -> 128x11x11
            #                 +
# 192x11x11 -> 16x11x11 -> 32x11x11
            #                 +
            # 192x11x11 -> 32x11x11
            #                 =
            #             256x11x11
            Inception(256, 128, 128, 192, 32, 96, 64),
            # 256x11x11 -> 128x11x11
            #                 +
# 256x11x11 -> 128x11x11 -> 192x11x11
            #                 +
# 256x11x11 -> 32x11x11 -> 96x11x11
            #                 +
            # 256x11x11 -> 64x11x11
            #                 =
            #             480x11x11
            nn.MaxPool2d(3, 2)
            # 480x11x11 -> 480x5x5
        )
        
        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifer = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer(x)
        return x

def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype=np.float32) / 255
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)

train_set = CIFAR10("./data", train=True, transform=data_tf, download=False)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10("./data", train=False, transform=data_tf, download=False)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

net = GoogleNet(3, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train(net, train_data,test_data,20,optimizer,criterion)
summary(net.cuda(), (3, 96, 96))