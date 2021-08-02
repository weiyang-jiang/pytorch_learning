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

from torchvision.datasets import ImageFolder

from utils import train

train_img = ImageFolder("example_data/hymenoptera_data/train")
im, label = train_img[0]

def make_data_list(data_type):

    BASE_PATH = "example_data/hymenoptera_data"
    target_path = os.path.join(BASE_PATH, data_type)
    if not os.path.exists(os.path.join(target_path, f"{data_type}.txt")):
        categories_list = os.listdir(target_path)
        f = open(os.path.join(target_path, f"{data_type}.txt"), "w")
        for num, category in enumerate(categories_list):
            category_path = os.path.join(target_path, category)
            category_path_list = os.listdir(category_path)
            random.shuffle(category_path_list)
            data_list = []
            for index, name in enumerate(category_path_list):
                data_list.append(f"{os.path.join(category_path, name)},{num}\n")
                if index % 100 == 0 or index == len(category_path) - 1:
                    f.writelines(data_list)
                    data_list = []
        f.close()




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
        label = int(label.strip())
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)

def idtify(output):
    total = output.shape[0]
    classes = ["ants", "bees"]
    pred_label = output.max(1)[1]
    print(pred_label)
    result = classes[int(pred_label)]
    return result

if __name__ == '__main__':

    net = models.resnet50(pretrained=True)

    train_set = custom_dataset("example_data/hymenoptera_data/train/train.txt", transform=train_tf)
    test_set = custom_dataset("example_data/hymenoptera_data/val/val.txt", transform=test_tf)
    train_data = DataLoader(train_set, batch_size=1, shuffle=True)
    test_data = DataLoader(test_set, batch_size=1, shuffle=False)

    if os.path.exists("ants_bees.pth"):
        net.fc = nn.Linear(2048, 2)
        model = torch.load("ants_bees.pth")
        net.load_state_dict(model)
    else:
        for param in net.parameters():
            param.requires_grad = False
        net.fc = nn.Linear(2048, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.01, weight_decay=1e-4)
        train(net, train_data, test_data, 20, optimizer, criterion)
        torch.save(net.state_dict(), "ants_bees.pth")

    # Image.open("example_data/hymenoptera_data/train/bees/21399619_3e61e5bb6f.jpg")
    net.eval()
    im = test_tf("/home/weiyang/pytorach/pytorch_learning/pytroch卷积神经网络/example_data/hymenoptera_data/train/ants/6240329_72c01e663e.jpg")
    im = im.unsqueeze(dim=0).cuda()
    net = net.cuda()
    output = net(im)
    print(output)
    print(idtify(output))


