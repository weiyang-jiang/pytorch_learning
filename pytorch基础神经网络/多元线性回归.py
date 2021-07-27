import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

x_sample = np.arange(-3, 3.1, 0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3

plt.plot(x_sample, y_sample, label='real	curve')
plt.legend()

x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
x_train = torch.from_numpy(x_train).float()  # 转换格式
y_train = torch.from_numpy(y_sample).float().unsqueeze(1)  # 转换格式并且增加了一个为维度

w = Variable(torch.randn(3, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

x_train = Variable(x_train)
y_train = Variable(y_train)


def multi_linear(x):
    return torch.mm(x, w) + b

def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)

# y_pred = multi_linear(x_train)
# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting	curve',
#          color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real	curve', color='b')
# plt.legend()

# loss = get_loss(y_pred, y_train)
# print(loss)
for epoch in range(10000):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred, y_train)
    if w.grad != None:
        w.grad.zero_()
        b.grad.zero_()
    loss.backward()
    w.data = w.data - 1e-3 * w.grad.data
    b.data = b.data - 1e-3 * b.grad.data
    print("loss: {}, w: {}, b: {}".format(loss.item(), w.data.numpy().squeeze(), b.data.numpy()[0]))

