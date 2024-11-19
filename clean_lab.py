import os
import matplotlib.pyplot as plt
import numpy as np
import cleanlab
import pickle
import sklearn
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cleanlab import Datalab
import numpy as np
from deepcore import nets, datasets
from sklearn import metrics as mr

def l1_dist(x, y):
    x = torch.from_numpy(x).reshape(x.shape[0], -1)
    y = torch.from_numpy(y).reshape(x.shape[0], -1)
    x = x.float()
    y = y.float()
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
    return dist

def euclidean_dist(x, y):
    # x = np.reshape(x, -1)
    # y = np.reshape(y, -1)
    # dist = mr.mutual_info_score(x, y)
    print(x.shape)
    x = torch.from_numpy(x).reshape(x.shape[0], -1)
    y = torch.from_numpy(y).reshape(y.shape[0], -1)
    print(x.shape)
    x = x.float()
    y = y.float()
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
    return dist

if __name__ == '__main__':



    # list1 = np.array([10, 20, 30, 40, 50])
    # list2 = np.array([30, 40, 60])
    #
    # indices = np.where(np.isin(list1, list2))[0]
    #
    #
    # print("元素在list1中的索引：", indices)
    # list1[indices] = list1[indices] * 0.5
    # print(list1)
    # exit(0)

    # data_train = MNIST('./data',
    #                    download=False,
    #                    transform=transforms.Compose([
    #                        transforms.Resize((32, 32)),
    #                        transforms.ToTensor()]))
    # data_test = MNIST('./data',
    #                   train=False,
    #                   download=False,
    #                   transform=transforms.Compose([
    #                       transforms.Resize((32, 32)),
    #                       transforms.ToTensor()]))
    # data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    # data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
    # print(type(data_train))
    # data = data_train.data
    # labels = data_train.targets
    data_path = 'data'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__["CIFAR10"](data_path)

    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False,
                                              num_workers=4, pin_memory=True)
    data = dst_train.data
    labels = dst_train.targets

    folder_name = 'test_data'
    os.makedirs(folder_name, exist_ok=True)

    error_file = os.path.join(folder_name, 'error_CIFAR10.pkl')
    error_index = []
    with open(error_file, 'rb') as file:
        error_index = pickle.load(file)
    print(dst_train.class_to_idx)
    # 构造文件路径
    for id in range(10):
        id = 0
        file_path = os.path.join(folder_name, 'CIFAR10_{}.pkl'.format(id))
        with open(file_path, 'rb') as file:
            loaded_array = pickle.load(file)
            start = 0
            size = 1000
            dist = euclidean_dist(data[loaded_array[start:start+size]], data[loaded_array[0:1000]])
            # print("dis:", dist)
            print("dis:", dist.sum())
            print("sum:", dist.sum()/(size * 1000 - size))
            break
            plt.imshow(data[loaded_array[-2]])
            plt.title(f'Label: {labels[loaded_array[-2]]}')
            plt.axis('off')
            # plt.show()

            plt.imshow(data[loaded_array[0]])
            plt.title(f'Label: {labels[loaded_array[0]]}')
            plt.axis('off')
            # plt.show()

            # print("worst-best dis: ", mr.mutual_info_score(data[loaded_array[0]], data[loaded_array[-1]]))

            print("worst-best dis: ", euclidean_dist(data[loaded_array[0]], data[loaded_array[-1]]))
            indices = [i for i, x in enumerate(loaded_array) if x in error_index]
            print(indices)
            index_list = loaded_array[indices]
            print(index_list)
            for idx in index_list:
                plt.imshow(data[idx])
                plt.title(f'Label: {labels[idx]}')
                plt.axis('off')
                print("worst-error dis: ", euclidean_dist(data[loaded_array[0]], data[idx]))
                print("best-error dis: ", euclidean_dist(data[loaded_array[-1]], data[idx]))
                # plt.show()
            final = loaded_array[~np.isin(loaded_array, error_index)]
        break
            # print(final[:100])

    # channel = 1
    # im_size = (28, 28)
    # num_classes = 10
    # network = nets.__dict__["LeNet"](channel, num_classes, im_size)  # cleanlab works with **any classifier**. Yup, you can use PyTorch/TensorFlow/OpenAI/XGBoost/etc.
    # cl = cleanlab.classification.CleanLearning(network)
    #
    # # cleanlab finds data and label issues in **any dataset**... in ONE line of code!
    # label_issues = cl.find_label_issues(data, labels)
    #
    # # cleanlab trains a robust version of your model that works more reliably with noisy data.
    # cl.fit(data, labels)
    #
    # # cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
    # # cl.predict(test_data)
    #
    # # A universal data-centric AI tool, cleanlab quantifies class-level issues and overall data quality, for any dataset.
    # cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)