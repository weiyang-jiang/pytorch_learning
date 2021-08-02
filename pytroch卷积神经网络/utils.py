from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

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
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
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
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype=np.float32) /255
    x = (x - 0.5) / 0.5
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x)
    return x

def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(120),  # 调整图片的大小
        tfs.RandomHorizontalFlip(),  # 让图片进行水平翻转
        tfs.RandomResizedCrop((96, 96)),  # 随机截取96x96的区域作为输入值
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),  # 随机从0.5-1.5之间亮度变化
        # 说明中的[max(0, 1 - brightness), 1 + brightness]就是 [0.5 , 1.5]
        # hue_factor从[-hue, hue]中随机采样产生，其值应当满足0<= hue <= 0.5或-0.5 <= min <= max <= 0.5
        tfs.ToTensor(), # 这个是专门用来转换图片的，让【0-255】的图片转换到【0-1】的区间
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 这个方法是用来将【0-1】区间的图片转换到【-1到1】之间正太分部的
        # 第一个值表示三通道的均值，第二值表示三通道的方差。
        # 运算为 x = (x - mean(x))/ std(x)
    ])
    x = im_aug(x)
    return x

# 训练集要做多层转换， 但是测试集就保持正常水平， 为了模拟真实场景
def test_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize((96, 96)),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    x = im_aug(x)
    return x

# 自制数据集
class custom_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform
        with open(txt_path, "r") as file:
            lines = file.readlines()

        self.img_list = [i.split(",")[0] for i in lines]
        self.label_list = [i.split(",")[1] for i in lines]

    def __getitem__(self, item):
        img = self.img_list[item]
        label = self.label_list[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)

def load_data(dataset, batch_size, shuffle):
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    txt_dataset = custom_dataset("example_data/train.txt")
