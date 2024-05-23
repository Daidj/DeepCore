from typing import Any, Callable, Optional
import numpy as np
import functools
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torch import tensor, long, nn


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
        self.classes = [str(c) for c in range(2)]
        self.transform = transform
        self.target_transform = target_transform
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
            # 有效样本496
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


def MEDICAL(data_path):
    channel = 3
    im_size = (700, 700)
    num_classes = 2
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    folder_name = os.path.join(data_path, 'MEDICAL')
    dst_train = MedicalDataset(root=folder_name, train=True, transform=transform)
    dst_test = MedicalDataset(root=folder_name, train=False, transform=transform)
    # dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    # dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
