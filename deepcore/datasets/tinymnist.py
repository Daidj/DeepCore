import torch
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset

class TINYMNISTDataSet(Dataset):
    def __init__(self, root: str, transform):
        self.root_dir = root
        self.classes = [x for x in range(10)]
        self.num_classes = len(self.classes)  # 类别数
        # self.targets = []
        dst_train = datasets.MNIST(root, train=True, download=True, transform=transform)
        n_train = len(dst_train)
        indices = np.array([], dtype=np.int64)
        for c in range(self.num_classes):
            class_index = np.arange(n_train)[dst_train.targets == c]
            indices = np.append(indices, class_index[0:200])
        self.data = dst_train.data[indices]
        self.targets = dst_train.targets[indices]
        self.transform = transform

        print("TINY MNIST Dataset init")

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def TINYMNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = TINYMNISTDataSet(data_path, transform)

    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


if __name__ == '__main__':
    TINYMNIST('/home/sample_selection/data/')
    print('hello')
