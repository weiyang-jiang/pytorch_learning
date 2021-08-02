import torch
import os
import random
from torch.utils.data import DataLoader, Dataset

def get_data_list():
    f_train = open("train.txt", "w")
    f_test = open("test.txt", "w")
    categories = os.listdir("train")
    for category in categories:
        c_list_path = os.path.join("train", category)
        c_list = os.listdir(c_list_path)
        random.shuffle(c_list)
        label = int(category.split("c")[1])
        epoch_list = []
        for index, c in enumerate(c_list):
            data_path = os.path.join(c_list_path, c)
            epoch_list.append(f"{data_path},{label}\n")
            if index <= 499 and index % 200 == 0 or index == 499:
                f_test.writelines(epoch_list)
                epoch_list = []
            elif index > 499 and index % 200 ==0 or index == len(c_list) - 1:
                f_train.writelines(epoch_list)
                epoch_list = []
    f_train.close()
    f_test.close()

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

if __name__ == '__main__':
    pass