from torchvision import datasets, transforms
from torch import tensor, long
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SubCIFAR100DataSet(Dataset):
    indexes = None

    def __init__(self, root: str, transform, index, divide_step=5):
        total_class_num = 100
        if SubCIFAR100DataSet.indexes is None:
            np.random.seed(100)
            array = np.arange(total_class_num)  # 生成 0 到 100 的数组
            np.random.shuffle(array)
            SubCIFAR100DataSet.indexes = array


        self.root_dir = root
        self.num_classes = total_class_num
        self.step_num = round(total_class_num/divide_step) # 类别数
        self.classes = [x for x in range(self.num_classes)]

        # self.targets = []
        dst_train = datasets.CIFAR100(root, train=True, download=True, transform=transform)
        n_train = len(dst_train)
        indices = np.array([], dtype=np.int64)
        step_class = SubCIFAR100DataSet.indexes[index*self.step_num:(index+1)*self.step_num]
        for c in step_class:
            class_index = np.arange(n_train)[dst_train.targets == c]
            indices = np.append(indices, class_index)
        self.data = dst_train.data[indices]
        self.targets = np.array(dst_train.targets)[indices]
        self.transform = transform


        print("Sub CIFAR100 Dataset init")

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

def SubCIFAR100(data_path, index=0, divide_step=5):
    channel = 3
    im_size = (32, 32)

    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = SubCIFAR100DataSet(data_path, transform, index, divide_step)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

if __name__ == '__main__':
    SubCIFAR100('/home/sample_selection/data/', index=4)
    print('hello')