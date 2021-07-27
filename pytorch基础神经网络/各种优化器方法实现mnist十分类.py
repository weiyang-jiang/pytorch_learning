import numpy as np
import torch
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

"""
经验之谈
对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值
SGD通常训练时间更长，容易陷入鞍点，但是在好的初始化和学习率调度方案的情况下，结果更可靠
如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。
Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果
————————————————
版权声明：本文为CSDN博主「ycszen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u012759136/article/details/52302426

"""

class optimizer(object):
    def __init__(self):
        """
        https://zhuanlan.zhihu.com/p/55150256
        https://blog.csdn.net/u012759136/article/details/52302426
        """
        pass

    def SGD_optimizer(self, parameters, lr):
        """
        随机梯度下降法，固定学习率，每经过一次反向传播就更新一次梯度值，
        由于需要考虑竖直和水平的梯度问题， 所以不能把学习率设置的太大，
        导致收敛会偏慢，但是这种也是很常用的优化算法

        选择合适的learning rate比较困难 ，学习率太低会收敛缓慢，学习率过高会使收敛时的波动过大
        所有参数都是用同样的learning rate
        SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点
        公式：
                param = param - lr * param.grad
        :param parameters: 网络参数
        :param lr: 学习率
        :return: None
        """
        for param in parameters:
            param.data = param.data - lr * param.grad.data

    def momentum(self, parameters, vs, lr, gamma):
        """
        动量法随机梯度下降：动量优化方法引入物理学中的动量思想，加速梯度下降，
        有Momentum和Nesterov两种算法。当我们将一个小球从山上滚下来，没
        有阻力时，它的动量会越来越大，但是如果遇到了阻力，速度就会变小，
        动量优化法就是借鉴此思想，使得梯度方向在不变的维度上，参数更新变快，
        梯度有所改变时，更新参数变慢，这样就能够加快收敛并且减少动荡。
        :param parameters:
        :param vs:
        :param lr:
        :param gamma:
        :return:
        """
        for param, v in zip(parameters, vs):
            v[:] = gamma * v + lr * param.grad.data
            param.data = param.data - v

    def sgd_adagrad(self, parameters, sqrs, lr):
        """
        从上式可以看出，梯度加速变量r为t时刻前梯度的平方和
        [公式] , 那么参数更新量 [公式] ，将 [公式]
        看成一个约束项regularizer. 在前期，梯度累计平方和比较小，
        也就是r相对较小，则约束项较大，这样就能够放大梯度,
        参数更新量变大; 随着迭代次数增多，梯度累计平方和也越来越大，
        即r也相对较大，则约束项变小，这样能够缩小梯度，参数更新量变小。

        缺点：

            仍需要手工设置一个全局学习率lr , 如果lr设置过大的话，
            会使regularizer过于敏感，对梯度的调节太大
            中后期，分母上梯度累加的平方和会越来越大，
            使得参数更新量趋近于0，使得训练提前结束，无法学习

        :param parameters:
        :param sqrs:
        :param lr:
        :return:
        """
        eps = 1e-10
        for param, sqr in zip(parameters, sqrs):
            sqr[:] = sqr + param.grad.data ** 2
            div = lr / torch.sqrt(sqr + eps) * param.grad.data
            param.data = param.data - div

    def rmsprop(self, parameters, sqrs, lr, alpha=0.9):
        """
        RMSProp算法修改了AdaGrad的梯度平方和累加为指数加权的移动平均，
        使得其在非凸设定下效果更好。设定参数：全局初始率 lr , 默认设为0.001; decay rate ,默认设置为0.9,一个极小的常量 ，通常为10e-6

        其实RMSprop依然依赖于全局学习率lr
        RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间
        适合处理非平稳目标(包括季节性和周期性)——对于RNN效果很好

        解决了训练后期不会出现学习率过低的情况
        :param parameters:
        :param sqrs:
        :param lr:
        :param alpha:
        :return:
        """
        eps = 1e-10
        for param, sqr in zip(parameters, sqrs):
            sqr[:] = alpha * sqr + (1 - alpha) * param.grad.data ** 2
            div = lr / torch.sqrt(sqr + eps) * param.grad.data
            param.data = param.data - div

    def adadelta(self, parameters, sqrs, deltas, rho=0.9):
        """
        Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值
        特点：

            训练初中期，加速效果不错，很快。
            训练后期，反复在局部最小值附近抖动。

        :param parameters:
        :param sqrs:
        :param deltas:
        :param rho:
        :return:
        """
        eps = 1e-6

        for param, sqr, delta in zip(parameters, sqrs, deltas):
            sqr[:] = rho * sqr + (1 - rho) * param.grad.data ** 2
            cur_delta = torch.sqrt(delta + eps) / torch.sqrt(sqr + eps) * param.grad.data
            delta[:] = rho * delta + (1 - rho) * cur_delta ** 2
            param.data = param.data - cur_delta

    def adam(self, parameters, vs, sqrs, lr, t, beta1=0.9, beta2=0.999):
        """
        特点：

            Adam梯度经过偏置校正后，每一次迭代学习率都有一个固定范围，使得参数比较平稳。
            结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
            为不同的参数计算不同的自适应学习率
            也适用于大多非凸优化问题——适用于大数据集和高维空间。


        :param parameters: 网络参数
        :param vs: 动量变量 ： 一维动量
        :param sqrs: 梯度元素平方的移动指数加权平均 ： 二维动量
        :param lr: 学习率
        :param t: 迭代次数
        :param beta1: 权重参数
        :param beta2: 权重参数
        :return:
        """
        eps = 1e-8

        for param, v, sqr in zip(parameters, vs, sqrs):
            v[:] = beta1 * v + (1 - beta1) * param.grad.data
            sqr[:] = beta2 * sqr + (1 - beta2) * param.grad.data ** 2
            v_hat = v / (1 - beta1 ** t)
            s_hat = sqr / (1 - beta2 ** t)
            param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)

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

# optim = torch.optim.SGD(MnNet.parameters(), lr=0.01)
optim = optimizer()

sqrs = []
deltas = []
for param in MnNet.parameters():
    sqrs.append(torch.zeros_like(param.data))
    deltas.append(torch.zeros_like(param.data))


for epoch in range(5):
    total_loss = 0
    train_acc = 0
    test_acc = 0
    for im, label in train_data:
        output = MnNet(im)
        loss = criterion(output, label)
        MnNet.zero_grad()
        loss.backward()
        optim.rmsprop(MnNet.parameters(), vs, lr=0.01)
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