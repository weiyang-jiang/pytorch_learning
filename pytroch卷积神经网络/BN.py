import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import mnist
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils import *


def simple_BN(x, gamma, beta):
    """
        1.求数据均值
        2.求数据方差
        3.数据进行标准化（个人认为称作正态化也可以）
        4.训练参数γ，β
        5.输出y通过γ与β的线性变换得到新的值

    :param x:
    :param gamma:
    :param beta:
    :return:
    """
    eps = 1e-5  # 定义一个阀值，防止梯度消失
    # 1.求数据均值
    x_mean = torch.mean(x, dim=0, keepdim=True)
    print(x_mean.shape)
    # 2.求数据方差
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    print(x_var.shape)
    # 3.数据进行标准化
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    print(x_hat.shape)
    # 4.训练参数γ，β
    out_put = gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)
    print(out_put.shape)
    return out_put


def Batchnorm(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True)
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    moving_momentum = torch.FloatTensor([moving_momentum])
    if torch.cuda.is_available():
        moving_momentum = moving_momentum.cuda()
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1 - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1 - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


class multi_network(nn.Module):
    def __init__(self):
        super(multi_network, self).__init__()
        self.layer1 = nn.Linear(784, 100)
        self.relu = nn.ReLU(True)
        self.layer2 = nn.Linear(100, 10)

        self.gamma = nn.Parameter(torch.randn(100))
        self.beta = nn.Parameter(torch.randn(100))

        self.moving_mean = Variable(torch.zeros(100))
        self.moving_var = Variable(torch.zeros(100))

    def forward(self, x, is_train=True):
        x = self.layer1(x)
        x = Batchnorm(x, self.gamma, self.beta, is_train, self.moving_mean, self.moving_var)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def data_tf_BN(x):
    x = np.array(x, dtype=np.float32) / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1))
    x = torch.from_numpy(x)
    return x


net = multi_network()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)

train_set = mnist.MNIST("./mnist_data", train=True, transform=data_tf_BN, download=False)
test_set = mnist.MNIST("./mnist_data", train=False, transform=data_tf_BN, download=False)

train_data = DataLoader(train_set, batch_size=32, shuffle=True)
test_data = DataLoader(test_set, batch_size=32, shuffle=False)

train(net, train_data, test_data, 20, optimizer, criterion)
