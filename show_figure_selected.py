import os
import shutil

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from deepcore import datasets
import matplotlib.pyplot as plt

from deepcore.methods import MMDCalculator, InfoCalculator, Individual


def show_figure(data, index, save_folder='test_data/', type='selected', text=''):
    img = data[index]
    # label = targets[selected[0]]
    # assert label == c
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    # height, width = img.shape

    plt.axis('off')
    plt.text(0.0, 0.0, '{}'.format(text),
             ha='center', va='bottom', fontsize=14)
    folder = os.path.join(save_folder, '{}'.format(type))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, '{}.png'.format(index)))
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data_path = 'data'

    dataset = 'TINYMNIST'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset](data_path)
    n_train = len(dst_train)
    best = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fraction = 0.7
    data = dst_train.data
    targets = dst_train.targets
    test_data_folder = '/home/sample_selection/test_data/'
    folder = os.path.join(test_data_folder, 'iter_MOEAD_100/multi_{}'.format(dataset))
    best_file_path = os.path.join(folder, 'best_{}.npy'.format(fraction))
    all_best_result = np.load(best_file_path)
    best_index = np.load(os.path.join(folder, 'best_index_{}.npy'.format(fraction)))
    best_result = all_best_result

    # calculators = []
    # for c in range(1):
    #     features_matrix = torch.load(
    #         os.path.join(folder, 'label_{}/features_matrix_{}_{}.pth'.format(c, fraction, dataset))).cpu()
    #     confidence = torch.load(os.path.join(folder, 'label_{}/importance_{}_{}.pth'.format(c, fraction, dataset)))
    #
    #     calculators.append(fitness_calculators)

    for c in range(0, 10):

        folder = os.path.join(test_data_folder, 'iter_100/ldea_{}'.format(c))
        features_matrix = torch.load(
            os.path.join(folder, 'features_matrix_{}_{}.pth'.format(fraction, dataset))).cpu().numpy()
        confidence = torch.load(os.path.join(folder, 'importance_{}_{}.pth'.format(fraction, dataset)))
        size = round(features_matrix.shape[0] * fraction)

        fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
                               InfoCalculator(features_matrix, confidence, size, device='cuda')]
        Individual.fitness_calculators = fitness_calculators
        best_i = Individual(features_matrix.shape[0], size, 2)

        save_folder = 'selection_data/{}_{}'.format(dataset, c)
        if os.path.exists(save_folder):

            shutil.rmtree(save_folder)

        all = np.arange(n_train)[dst_train.targets == c]
        common_elements = np.intersect1d(all, best_result)

        # 找到这些共同元素在 all 中的索引
        selected = np.where(np.isin(all, common_elements))[0]


        best_i.init(selected)
        f1_selected_list = fitness_calculators[0].origin_selected_fitness(best_i)
        f2_selected_list = fitness_calculators[1].origin_selected_fitness(best_i)
        f1_unselected_list = fitness_calculators[0].origin_unselected_fitness(best_i)
        f2_unselected_list = fitness_calculators[1].origin_unselected_fitness(best_i)
        f1_min = min(f1_selected_list.min(), f1_unselected_list.min())
        f1_max = max(f1_selected_list.max(), f1_unselected_list.max())
        f2_min = min(f2_selected_list.min(), f2_unselected_list.min())
        f2_max = max(f2_selected_list.max(), f2_unselected_list.max())
        f1_selected_list = (f1_selected_list - f1_min)/(f1_max-f1_min)
        f1_unselected_list = (f1_unselected_list - f1_min)/(f1_max-f1_min)
        f2_selected_list = (f2_selected_list - f2_min)/(f2_max-f2_min)
        f2_unselected_list = (f2_unselected_list - f2_min) / (f2_max - f2_min)
        selected_list = list(best_i.gene)
        for i in range(len(selected_list)):
            type = 'selected'
            index = all[selected_list[i]]
            text = 'f1={:.4f}, f2={:.4f}'.format(f1_selected_list[i], f2_selected_list[i])
            show_figure(data, index, save_folder=save_folder, type=type, text=text)
        unselected_list = list(best_i.unselected_gene)
        for i in range(len(unselected_list)):
            type = 'unselected'
            index = all[unselected_list[i]]
            text = 'f1={:.4f}, f2={:.4f}'.format(f1_unselected_list[i], f2_unselected_list[i])
            show_figure(data, index, save_folder=save_folder, type=type, text=text)
    print('hello world!')
