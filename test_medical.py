import math
from typing import Any, Callable, Optional, Tuple

import numpy as np
import functools

import torch
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

from deepcore import nets
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torch import tensor, long, nn

from utils import accuracy


def custom_sort(s1, s2):
    if s1[0] != s2[0]:
        return -1 if s1[0] < s2[0] else 1
    else:
        num1 = int(s1[1:-4])
        num2 = int(s2[1:-4])
        return -1 if num1 < num2 else 1


def read_chip(row, column, dataframe):
    labels = []
    for col in column:
        for r in row:
            if dataframe[r][col] == 1:
                labels.append(1)
            elif dataframe[r][col] == 0:
                labels.append(0)
            else:
                labels.append(-1)
    return labels


class MedicalDataset(Dataset):
    train_targets_format = {
        "芯片2": {
            "row": list(range(22, 38)),
            "column": [i for i in range(3, 10) if i % 2 != 0]
        },
        "芯片4": {
            "row": list(range(41, 53)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片5": {
            "row": list(range(56, 68)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片6": {
            "row": list(range(71, 83)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片7": {
            "row": list(range(86, 98)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片8": {
            "row": list(range(101, 113)),
            "column": [i for i in range(3, 16) if i % 2 != 0]
        }
    }

    test_targets_format = {
        "芯片1": {
            "row": list(range(2, 18)),
            "column": [i for i in range(3, 22) if i % 2 != 0]
        }
    }

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, ):
        self.root_dir = root
        self.train = train
        self.data: Any = []
        self.targets = []
        self.init_file_list()

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.target_transform = None
        print("Dataset init finished")

    def init_file_list(self):
        folder_name = self.root_dir
        targets_file_path = os.path.join(folder_name, 'chip_labels.xlsx')
        # 读取Excel文件
        excel_data = pd.read_excel(targets_file_path, sheet_name='总芯片', header=None)
        df = pd.DataFrame(excel_data)

        if self.train:
            # 读取图像数据
            for i in range(2, 9):
                if i == 3:
                    continue
                chip_folder = os.path.join(folder_name, 'chip_{}'.format(i))
                image_names = os.listdir(chip_folder)
                image_names.sort(key=functools.cmp_to_key(custom_sort))
                for name in image_names:
                    image_name = os.path.join(chip_folder, name)
                    image = Image.open(image_name)
                    # image_array = np.transpose(np.asarray(image), (2, 0, 1))
                    image_array = np.asarray(image)
                    self.data.append(image_array)
            # 读取标签
            for key, value in self.train_targets_format.items():
                labels = read_chip(row=value["row"], column=value["column"],
                                   dataframe=df.values)
                self.targets.extend(labels)
            self.targets = self.targets[5:len(self.targets) - 7]
            self.data = np.vstack(self.data).reshape(-1, 700, 700, 3)
            self.targets = np.array(self.targets)
            # 有效样本496，阳性样本41个，阴性样本455
            indices = np.where(np.array(self.targets) != -1)[0]
            self.data = self.data[indices]
            self.targets = self.targets[indices]


        else:
            # 读取图像
            chip_folder = os.path.join(folder_name, 'chip_1')
            image_names = os.listdir(chip_folder)
            image_names.sort(key=functools.cmp_to_key(custom_sort))
            for name in image_names:
                image_name = os.path.join(chip_folder, name)
                image = Image.open(image_name)
                image_array = np.asarray(image)
                # image_array = np.transpose(np.asarray(image), (2, 0, 1))
                self.data.append(image_array)
            # 读取标签
            for key, value in self.test_targets_format.items():
                labels = read_chip(row=value["row"], column=value["column"],
                                   dataframe=df.values)
                self.targets = labels
            self.data = np.vstack(self.data).reshape(-1, 700, 700, 3)
            self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
                """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        #
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    print(os.getcwd())
    train_dataset = MedicalDataset(root='./data/MEDICAL', train=True)
    test_dataset = MedicalDataset(root='./data/MEDICAL', train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)
    epochs = 10
    model = "LeNet"
    channel = 3
    im_size = (700, 700)
    num_classes = 2
    lr = 0.02
    min_lr = 0.0
    weight_decay = 5e-4
    scheduler_name = "CosineAnnealingLR"
    device = 'cuda'
    network = nets.__dict__[model](channel, num_classes, im_size).to(device)
    torch.cuda.set_device(0)
    network = nets.nets_utils.MyDataParallel(network, device_ids=[0])
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs,
                                                           eta_min=min_lr)
    total_step = len(train_loader)
    # for epoch in range(epochs):
    #     network.train()
    #
    #     for i, contents in enumerate(train_loader):
    #         optimizer.zero_grad()
    #
    #         target = contents[1].to(device)
    #         input = contents[0].to(device)
    #         # print(input.shape)
    #         # Compute output
    #         output = network(input)
    #         loss = criterion(output, target).mean()
    #
    #         # Measure accuracy and record loss
    #         # prec1 = accuracy(output.data, target, topk=(1,))[0]
    #         # losses.update(loss.data.item(), input.size(0))
    #         # top1.update(prec1.item(), input.size(0))
    #
    #         # Compute gradient and do SGD step
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         if i % 2 == 0:
    #             print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
    # test
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    exit(0)

    folder_name = './data/MEDICAL/'
    count = 0
    for i in range(1, 9):
        if i == 3:
            continue
        chip_folder = os.path.join(folder_name, 'chip_{}'.format(i))
        image_names = os.listdir(chip_folder)
        print(image_names)
        # image_names.sort(key=functools.cmp_to_key(lambda x, y: 1 if x[0] > y[0] else -1 or int(x[1:-4]) - int(y[1:-4])))

        image_names.sort(key=functools.cmp_to_key(custom_sort))
        print(image_names)
        print(len(image_names))
        for name in image_names:
            # 打开 PNG 文件
            image_name = os.path.join(chip_folder, image_names[0])
            image = Image.open(image_name)

            # 显示图像信息
            # print("图像格式:", image.format)
            # print("图像大小:", image.size)
            # print("图像模式:", image.mode)
            image_array = np.asarray(image)
            assert image_array.shape == (700, 700, 3)
            # 可以对图像进行进一步的操作，比如显示图像、裁剪、调整大小等
            # print(image_array)
            # plt.imshow(image)
            # plt.show()
            count += 1
        print(count)
        count = 0

    # 构造文件路径
    file_path = os.path.join(folder_name, 'chip_labels.xlsx')

    # 读取Excel文件
    data = pd.read_excel(file_path, sheet_name='总芯片', header=None)
    df = pd.DataFrame(data)
    # 芯片2缺少前5个数据，芯片8缺少后七个数据
    data_format = {
        "芯片1": {
            "row": list(range(2, 18)),
            "column": [i for i in range(3, 22) if i % 2 != 0]
        },
        "芯片2": {
            "row": list(range(22, 38)),
            "column": [i for i in range(3, 10) if i % 2 != 0]
        },
        "芯片4": {
            "row": list(range(41, 53)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片5": {
            "row": list(range(56, 68)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片6": {
            "row": list(range(71, 83)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片7": {
            "row": list(range(86, 98)),
            "column": [i for i in range(3, 18) if i % 2 != 0]
        },
        "芯片8": {
            "row": list(range(101, 113)),
            "column": [i for i in range(3, 16) if i % 2 != 0]
        }
    }
    # test_labels = read_chip(row=data_format["芯片1"]["row"], column=data_format["芯片1"]["column"], dataframe=df.values)
    test_labels = []
    train_labels = []
    for key, value in data_format.items():
        labels = read_chip(row=value["row"], column=value["column"],
                           dataframe=df.values)
        if key == "芯片1":
            test_labels = labels
        else:
            train_labels.extend(labels)
    # 打印读取的数据
    # print(df)
    # 训练集532个样本
    # 测试集160个样本
    print(len(train_labels))
    print(len(test_labels))
    print(train_labels)
    # for i in range(len(test_labels)):
    #     if test_labels[i] == 1:
    #         print(i)

    model = "LeNet"
    channel = 3
    im_size = (700, 700)
    num_classes = 2
    exit(0)
    device = 'cuda'
    network = nets.__dict__[model](channel, num_classes, im_size).to(device)
