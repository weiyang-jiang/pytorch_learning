import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

torch.manual_seed(2017)
with open('./data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

x_real = [i[0] for i in data]
y_real = [i[1] for i in data]
label = ["red" if i[2] == 1 else "blue" for i in data]

np_data = np.array(data, dtype=np.float32)
x_train = Variable(torch.from_numpy(np_data[:, 0:2]))
y_train = Variable(torch.from_numpy(np_data[:, -1]).unsqueeze(1))

num = 3
w = torch.nn.Parameter(torch.randn(2, num))
b = torch.nn.Parameter(torch.zeros(num))
b1 = torch.nn.Parameter(torch.randn(num, 1))

def func(x, w, b):
    return torch.mm((torch.mm(x, w) + b), b1)


optimizer = optim.SGD([w, b, b1], lr=0.01)

criterion = torch.nn.BCEWithLogitsLoss()



for epoch in range(10000):
    y_pred = func(x_train, w, b)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mask = y_pred.ge(0.5).float()  # ge函数是和0.5比较大小，如果大于0.5输出1，小于0.5输出0
    acc = sum([int(i) for i in (mask == y_train).squeeze(1).numpy()]) / y_train.shape[0]
    if (epoch + 1) % 200 == 0:
        print('epoch:	{},	Loss:	{:.5f},	Acc:	{:.5f}'.format(epoch + 1, loss, acc))

plot_x, plot_y = np.mgrid[0.2:1:0.01, 0.2:1:0.01]
grid = np.c_[plot_x.ravel(), plot_y.ravel()]  # 拉直

grid_ = torch.from_numpy(grid).to(torch.float32)
y_pre = torch.sigmoid(func(grid_, w, b)).detach().numpy().squeeze(1)

x_list = []
y_list = []
for index, y_data in enumerate(y_pre):
    if round(y_data, 2) == 0.5:
        x_list.append(grid[index, 0])
        y_list.append(grid[index, 1])

plt.plot(x_list, y_list, 'g', label='cutting line')
plt.scatter(x_real, y_real, c=label)
plt.legend(loc='best')
plt.show()
