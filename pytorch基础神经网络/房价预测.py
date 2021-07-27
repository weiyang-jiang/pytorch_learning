import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import os

MODEL_PATH = "FangNet.pth"
train = pd.read_csv("./dataset/train.csv")
test = pd.read_csv("./dataset/test.csv")

print(train.head())
print('一共有 {} 个训练集样本'.format(train.shape[0]))
print('一共有 {} 个测试集样本'.format(test.shape[0]))

all_features = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

# 取出除了文字信息的数值信息的索引名称
numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index # 取出所有的数值特征

# 减去均值，除以方差,这是一种归一化的数据标准化处理方法
all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean())
                                                                 / (x.std()))

# 将含有文字信息的数据进行独热码编写
all_features = pd.get_dummies(all_features, dummy_na=True)

# 将其中数据为null的地方以数据平均数进行填充
all_features = all_features.fillna(all_features.mean())

num_train = train.shape[0]

"""
这种数据转换思想很重要，房价和很多因素都有关， 将这些因素都用数值表现出来然后就可以进行，然后将其转换为矩阵shape(331, 1)
"""

train_features = all_features[:num_train].values.astype(np.float32)
test_features = all_features[num_train:].values.astype(np.float32)


train_labels = train.SalePrice.values.reshape(-1, 1).astype(np.float32)
test_labels = test.SalePrice.values.reshape(-1, 1).astype(np.float32)


# 打包特征值和标签值
def get_data(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)

train_data = get_data(train_features, train_labels, batch_size=64, shuffle=True)
test_data = get_data(test_features, test_labels, batch_size=1, shuffle=False)

class FangNet(nn.Module):
    def __init__(self, input_num, output_num):
        super(FangNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_num, 64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, output_num),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

Fang = FangNet(331, 1)

# 经过修改之后的损失函数
def get_rmse_log(model, feature, label):
    model.eval()
    mse_loss = nn.MSELoss()
    pred = model(feature)
    clipped_pred = torch.clamp(pred, 1, float('inf'))
    rmse = torch.sqrt(mse_loss(clipped_pred.log(), label.log()))
    return rmse


optimizer = torch.optim.Adam(Fang.parameters(), lr=0.01)
mse_loss = nn.MSELoss()

if not os.path.exists(MODEL_PATH):
    for epoch in range(2000):
        running_loss = 0
        for data, label in train_data:
            output = Fang(data)
            # print(output)
            # clipped_pred = torch.clamp(output, 10, float('inf'))
            # print(clipped_pred)
            loss = torch.sqrt(mse_loss(output, label.log()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
        print("epoch: {} loss: {}".format(epoch, running_loss / len(train_data)))
    torch.save(Fang.state_dict(), "FangNet.pth")
else:
    Fang.load_state_dict(torch.load(MODEL_PATH))

price_pre_list = []
price_real_list = []
for test_data, test_label in test_data:
    pred = Fang(test_data)
    price_pre = torch.exp(pred).data.reshape(-1).numpy().mean()
    price_pre_list.append(price_pre)
    price_real = test_label.data.reshape(-1).numpy().mean()
    price_real_list.append(price_real)

x = np.array(range(len(price_pre_list)))
plt.plot(x, price_pre_list, label="pre")
plt.plot(x, price_real_list, label="real")
plt.legend()
plt.show()