import os
import numpy as np
import torch

from deepcore import datasets
import matplotlib.pyplot as plt


def show_figure(data, index, save_folder='test_data/', text='selected'):
    img = data[index]
    # label = targets[selected[0]]
    # assert label == c
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    height, width, _ = img.shape

    plt.axis('off')
    plt.text(-width*0.05, -height*0.01, '{}:{}'.format(text, index),
         ha='center', va='bottom', fontsize=14)
    folder = os.path.join(save_folder, '{}'.format(text))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, '{}.png'.format(index)))
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data_path = 'data'

    dataset = 'CIFAR10'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset](data_path)
    n_train = len(dst_train)
    for c in range(10):
        folder = 'test_data/{}'.format(c)
        best_file_path = os.path.join(folder, 'best_{}.npy'.format(dataset))
        all = np.arange(n_train)[dst_train.targets == c]
        selected = np.load(best_file_path)
        unselected = np.setdiff1d(all, selected)
        print(selected.shape)
        print(unselected.shape)
        print('class: ', class_names[c])
        continue
        data = dst_train.data
        targets = dst_train.targets

        # for i in range(2):
        #     index = np.random.choice(selected)
        #     show_figure(data, index)
        # features_matrix = torch.load(os.path.join('test_data/features_matrix_{}.pt'.format(c)))
        importance = torch.load('test_data/importance_{}.pt'.format(c))
        _, sorted_indices = torch.sort(importance)
        for i in range(20):
            good = all[sorted_indices[i].item()]
            bad = all[sorted_indices[-i-1].item()]
            is_selected = np.any(selected == good)
            is_not_selected = np.any(unselected == good)
            if is_selected:
                text = 'selected'
            elif is_not_selected:
                text = 'unselected'
            else:
                text = 'error'
            show_figure(data, good, save_folder=folder, text=text)
            is_selected = np.any(selected == bad)
            is_not_selected = np.any(unselected == bad)
            if is_selected:
                text = 'selected'
            elif is_not_selected:
                text = 'unselected'
            else:
                text = 'error'
            show_figure(data, bad, save_folder=folder, text=text)
        continue

        # 困难：0, 1, 3 相似：-1，未知：-2, 4,5
        # 相似：-3，-4, -5 未知：-6, -7，-5

        # 困难：0，2，3，4 相似：-2， 未知：1
        # 相似：-1，未知：-3，-4
        best = sorted_indices[0].item()
        best = all[best]
        is_selected = np.any(selected == best)
        is_not_selected = np.any(unselected == best)
        if is_selected:
            text = 'selected'
        elif is_not_selected:
            text = 'selected'
        else:
            text = 'error'
        show_figure(data, best, text=text)
        worst = sorted_indices[-1].item()
        worst = all[worst]
        is_selected = np.any(selected == worst)
        is_not_selected = np.any(unselected == worst)
        show_figure(data, worst, text='unselected')
        # for i in range(1):
        #     index = np.random.choice(selected)
        #     show_figure(data, index, text='selected')
        #     index_in_class = np.where(all == index)[0][0]
        # show_figure(data, unselected[0])

    print('hello world!')
