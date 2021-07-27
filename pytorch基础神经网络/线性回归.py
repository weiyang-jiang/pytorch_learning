import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2017)

# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = np.arange(1, 10, 1)
y_train = x_train * 10 + 1
plt.scatter(x_train, y_train)
plt.show()

# 转换为tensor格式
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 设置梯度下降参数
w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

x_train = Variable(x_train)
y_train = Variable(y_train)


# 构建参数模型
def linear_model(x):
    return x * w + b


def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)

def plot_img(y_pre):
    plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
    plt.plot(x_train.data.numpy(), y_pre.data.numpy(), 'ro', label='estimated')
    plt.legend()
    plt.show()

# y_ = linear_model(x_train)
# loss = get_loss(y_, y_train)
# loss.backward()


for epoch in range(1000):
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)

    if w.grad != None:
        w.grad.zero_()  # 梯度清零，这个只对tensor使用
        b.grad.zero_()  #

    loss.backward()

    w.data = w.data - 0.02 * w.grad.data  # 更新参数
    b.data = b.data - 0.02 * b.grad.data  #
    # print('epoch:	{},	loss:	{}'.format(epoch, loss.item()))
    print("w: {}, b: {}".format(w.data.numpy()[0], b.data.numpy()[0]))
    # plot_img(y_)





