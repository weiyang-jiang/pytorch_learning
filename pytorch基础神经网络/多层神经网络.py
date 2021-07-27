import os

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim


def plot_decision_boundary(x, y):
    #	Set	min	and	max	values	and	give	it	some	padding
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    #	Generate	a	grid	of	points	with	distance	h	between	them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #	Predict	the	function	value	for	the	whole	grid
    Z = plot_net(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #	Plot	the	contour	and	training	examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)


np.random.seed(1)
m = 400  # 样本数量
N = int(m / 2)  # 每一类的点的个数， 二分类
D = 2  # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8')  #
a = 4

for j in range(2):
    ix = range(N * j, N * (j + 1))
    t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
    r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
    x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

x = torch.from_numpy(x / np.max(x)).float()
y = torch.from_numpy(y / np.max(y)).float()


class weiyangNet(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(weiyangNet, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(4, 10)
        self.layer4 = nn.Tanh()
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Tanh()
        self.layer7 = nn.Linear(10, 4)
        self.layer8 = nn.Tanh()
        self.layer9 = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x



weiyang = weiyangNet(2, 4, 1)


if os.path.exists("./weiyang.pth"):
    model = torch.load('./weiyang.pth')
    weiyang.load_state_dict(model)

else:
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(weiyang.parameters(), lr=0.1)
    for epoch in range(10001):
        output = weiyang(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mask = output.ge(0.5).float()
        acc = sum([int(i) for i in (mask == y).squeeze(1).numpy()]) / y.shape[0]
        # break
        if (epoch + 1) % 1000 == 0:
            print("epoch: {} loss :{} acc: {}".format(epoch + 1, loss.data, acc))
    torch.save(weiyang.state_dict(), './weiyang.pth')


def plot_net(x):
    out = torch.sigmoid(weiyang(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out


plot_decision_boundary(x.numpy(), y.numpy())
plt.title('sequential')
plt.show()
