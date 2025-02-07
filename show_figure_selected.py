import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from deepcore import datasets
import matplotlib.pyplot as plt


def show_figure(data, index, save_folder='test_data/', text='selected'):
    img = data[index]
    # label = targets[selected[0]]
    # assert label == c
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    # height, width, _ = img.shape
    #
    # plt.axis('off')
    # plt.text(-width * 0.05, -height * 0.01, '{}:{}'.format(text, index),
    #          ha='center', va='bottom', fontsize=14)
    folder = os.path.join(save_folder, '{}'.format(text))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, '{}.png'.format(index)))
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data_path = '/home/sample2/data'

    dataset = 'CIFAR100'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset](data_path)
    n_train = len(dst_train)
    best = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fraction = 0.5
    # plt.figure(figsize=(8, 6))
    data = dst_train.data
    targets = dst_train.targets
    folder = 'test_data/multi_{}'.format(dataset)
    best_file_path = os.path.join(folder, 'best_{}.npy'.format(fraction))
    all_best_result = np.load(best_file_path)
    best_index = np.load(os.path.join(folder, 'best_index_{}.npy'.format(fraction)))
    best_result = all_best_result[best_index[0]]

    for c in range(10):

        folder = 'test_data/multi_{}'.format(fraction)
        features_matrix = torch.load(
            os.path.join(folder, 'features_matrix_{}_{}.pth'.format(fraction, dataset))).cpu().numpy()
        confidence = torch.load(os.path.join(folder, 'importance_{}_{}.pth'.format(fraction, dataset)))

        save_folder = 'selection_data/{}_{}'.format(dataset, c)
        all = np.arange(n_train)[dst_train.targets == c]
        selected = np.load(best_file_path)
        unselected = np.setdiff1d(all, selected)

        for i in all:
            if i in best_result:
                is_selected = True
            else:
                is_selected = False
            if is_selected:
                text = 'selected'
            else:
                text = 'unselected'
            show_figure(data, i, save_folder=save_folder, text=text)
    print('hello world!')
