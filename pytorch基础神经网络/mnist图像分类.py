import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader



def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  #
    x = x.reshape((-1,))  #
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
a_data, a_label = train_set[0]
# a_data = np.array(a_data, dtype=np.float32)


# 使用迭代器，这样可以控制一次性喂入神经网络的batch_size
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


class MnistNet(nn.Module):
    def __init__(self, input_num, output_num):
        super(MnistNet, self).__init__()
        self.layer1 = nn.Linear(input_num, 400)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(400, 200)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(200, 100)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(100, output_num)
        # self.layer8 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        # x = self.layer8(x)
        return x

MnNet = MnistNet(28*28, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MnNet.parameters(), lr=0.1)


for epoch in range(20):
    total_loss = 0
    train_acc = 0
    test_acc = 0
    for im, label in train_data:
        output = MnNet(im)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        _, pred = output.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        train_acc += acc



    for im_test, label_test in test_data:
        output_test = MnNet(im_test)
        pred_test = output_test.max(1)[1]
        num_correct_test = (pred_test == label_test).sum().data
        acc_test = num_correct_test / im_test.shape[0]
        test_acc += acc_test

    epoch_loss = total_loss / len(train_data)
    epoch_train_acc = train_acc / len(train_data)
    epoch_test_acc = test_acc / len(test_data)
    print("loss: {}, train_acc: {}, test_acc: {}".format(epoch_loss, epoch_train_acc, epoch_test_acc))