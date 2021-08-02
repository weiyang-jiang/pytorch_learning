import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchsummary import summary
from datetime import datetime
from torch.utils.data import DataLoader


class AlexNet(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        # 第一层是5*5的卷积，输入的通道数为3， 输出通道数为64， 步长为1， 没有填充

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5)),  # N=(W-F+2P)/S+1=(32-5+2x0)/1+1= 28 (64x28x28)
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2)  # 池化是3*3的池化核， 步长为2， 没有填充
            # 当计算尺寸不被整除时，卷积向下取整，池化向上取整
            # N=(W-F+2P)/S+1=(28-3+2x0)/2+1=13 (64x13x13)
        )

        # 第二层是5*5的卷积， 输入的channels是64， 输出依然为64， 步长为1， 没有填充

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3)), # N=(W-F+2P)/S+1=(13-3+2x0)/1+1= 11 (64x11x11)
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2)  # 池化是3*3的池化核， 步长为2， 没有填充
            # N=(W-F+2P)/S+1=(11-3+2x0)/2+1= 5 (64x5x5)
        )

        # 这是一个全连接层，输入是1024， 输出是384
        self.fc1 = nn.Sequential(
            nn.Linear(num_input, 384),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(192, num_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x




def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        """
        model.train():
        在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
        
        model.eval():
        测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。

        """
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            with torch.no_grad():
                for im, label in valid_data:
                    if torch.cuda.is_available():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                    else:
                        im = Variable(im)
                        label = Variable(label)
                    output = net(im)
                    loss = criterion(output, label)
                    valid_loss += loss.item()
                    valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


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
net = AlexNet(1600, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()
train(net, train_data, test_data, num_epochs=10, optimizer=optimizer, criterion=criterion)
