from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms as tfs
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from create_data import *
import shutil
import glob

classes = ["安全驾驶", "右手打字","右手打电话","左手打字","左手打电话","调收音机","喝饮料","那后面的东西","整理头发和化妆","和其他乘客说话"]

def train_tf(x):
    im = Image.open(x)
    im_aug = tfs.Compose([
        # tfs.Resize(224),  # 调整图片的大小

        tfs.RandomResizedCrop(224),  # 随机截取96x96的区域作为输入值
        tfs.RandomHorizontalFlip(),  # 让图片进行水平翻转
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),  # 随机从0.5-1.5之间亮度变化
        # 说明中的[max(0, 1 - brightness), 1 + brightness]就是 [0.5 , 1.5]
        # hue_factor从[-hue, hue]中随机采样产生，其值应当满足0<= hue <= 0.5或-0.5 <= min <= max <= 0.5
        tfs.ToTensor(), # 这个是专门用来转换图片的，让【0-255】的图片转换到【0-1】的区间
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 这个方法是用来将【0-1】区间的图片转换到【-1到1】之间正太分部的
        # 第一个值表示三通道的均值，第二值表示三通道的方差。
        # 运算为 x = (x - mean(x))/ std(x)
    ])
    im = im_aug(im)
    return im

# 训练集要做多层转换， 但是测试集就保持正常水平， 为了模拟真实场景
def test_tf(x):
    im = Image.open(x)
    im_aug = tfs.Compose([
        tfs.Resize((224, 224)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = im_aug(im)
    return im

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    train_loss_list = []
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        if epoch == 20:
            set_lr(optimizer, lr=0.01)
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
        train_loss_list.append(train_loss/len(train_data))
    plt.plot(train_loss_list)
    plt.show()

path = "driver.pth"

def idtify(output):

    pred_label = output.max(1)[1]
    result = classes[int(pred_label)]
    return result, int(pred_label)


def get_result(net, txt_path):
    net.eval()
    test_path = os.listdir("test")
    im_data = []
    net = net.cuda()
    file = open(txt_path, "w")
    for index, im_path in enumerate(tqdm(test_path)):
        im = test_tf(os.path.join("test", im_path))
        im = im.unsqueeze(dim=0).cuda()
        output = net(im)
        re, label = idtify(output)
        im_data.append(f"{im_path},{label},{re}\n")
        if index % 200 == 0 or index == len(test_path) - 1:
            file.writelines(im_data)
            im_data = []
    file.close()

def seprarate(txt_path):
    file = open(txt_path, "r")
    datas = file.readlines()
    for c in classes:
        os.mkdir(f"test/{c}")
    for data in tqdm(datas):
        data = data.strip()
        path = os.path.join("test", data.split(",")[0])
        re = data.split(",")[2]
        shutil.move(path, os.path.join("test", re))

if __name__ == '__main__':
    net = models.resnet50(pretrained=True)

    train_set = custom_dataset("train.txt", transform=train_tf)
    test_set = custom_dataset("test.txt", transform=test_tf)

    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=64, shuffle=False)

    if os.path.exists(path):
        net.fc = nn.Linear(2048, 10)
        model = torch.load(path)
        net.load_state_dict(model)
    else:
        net.fc = nn.Linear(2048, 10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
        train(net, train_data, test_data, 40, optimizer, criterion)
        torch.save(net.state_dict(), path)
    if not os.path.exists("result.txt"):
        get_result(net, "result.txt")
    seprarate("result.txt")
