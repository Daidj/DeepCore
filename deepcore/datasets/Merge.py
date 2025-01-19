import torch
from torchvision import datasets, transforms
from torch import tensor, long, cat
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from deepcore.datasets import SubCIFAR100


class MergeDataSet(Dataset):

    def __init__(self, root: str, transform, dataset, index):
        total_class_num = 100

        self.root_dir = root
        self.num_classes = total_class_num
        self.classes = [x for x in range(self.num_classes)]

        self.data = dataset.data[index]
        self.targets = dataset.targets[index]
        self.transform = transform


        print("Merge Dataset init")

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def merge(self, dataset, index):
        self.data = np.vstack((self.data, dataset.data[index]))
        self.targets = cat((self.targets, dataset.targets[index]))


def Merge(data_path, dataset, index):
    channel = 3
    im_size = (32, 32)

    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = MergeDataSet(data_path, transform, dataset, index)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

if __name__ == '__main__':
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = SubCIFAR100('/home/sample_selection/data/', index=0)
    channel, im_size, num_classes, class_names, mean, std, dt, dst_test = Merge('/home/sample_selection/data/', dst_train, np.array([0]))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = SubCIFAR100('/home/sample_selection/data/', index=1)
    dst_subset = torch.utils.data.Subset(dst_train, np.array([0, 2, 4]))
    dt.merge(dst_train, np.array([1, 3, 5, 7]))
    print('hello')