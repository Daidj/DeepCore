# from deepcore.methods.moea_d_ldea import SubProblems
import math
import os
import random
import pandas as pds

import numpy
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from deepcore import datasets
from deepcore.methods import MMDCalculator, k_center_greedy, euclidean_dist_for_batch, euclidean_dist, InfoCalculator
from deepcore.methods.micro import micro
from deepcore.methods.moea_d_ldea_new import Individual, plot_nested_list
from deepcore.methods.two_stage_search import two_stage_search
from deepcore.methods.self_adaptation_search import self_adaptation_search
from kl import run_test
from mmd_algorithm import MMD
from pymoo.indicators.hv import HV


def get_best_in_solution(best_solution, fraction=None, selected=None):
    solution_num = 5
    if fraction is None:
        step = 1.0 / (solution_num - 1)
        first = random.randint(0, solution_num - 1)
        fraction = torch.tensor([first * step, 1.0 - first * step])
        print('fraction:', fraction)
    else:
        fraction = torch.tensor(fraction)
    fitness_front = [p.fitness for p in best_solution]
    front_tensor = torch.tensor(fitness_front)
    # greedy_best = torch.tensor(self.greedy_best_fitness_points)
    best_point = torch.min(front_tensor, dim=0).values
    worst_point = torch.max(front_tensor, dim=0).values
    scores = (front_tensor - best_point) / (worst_point - best_point + 1e-8)

    fraction_matrix = fraction.unsqueeze(0).repeat(scores.size(0), 1)
    scores = torch.sum(scores * fraction_matrix, dim=1)
    if selected is None:
        best = torch.argmin(scores)
    else:
        selected = set([i.item() for i in selected])
        index = 0
        sorted_list = torch.argsort(scores)
        while index < len(scores) and sorted_list[index].item() in selected:
            index += 1
        best = sorted_list[index]
    return best


def get_multi_best_solution(best_solution):
    solution_num = 5
    best_list = []
    step = 1.0 / (solution_num - 1)
    for i in range(solution_num):
        fraction = [max(i * step, 0.00001), max(1.0 - i * step, 0.00001)]
        best_list.append(get_best_in_solution(best_solution, fraction, best_list))
    return best_list

if __name__ == '__main__':
    data_path = 'data'

    dataset = 'TINYMNIST'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset](data_path)
    n_train = len(dst_train)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fraction = 0.7
    plt.figure(figsize=(8, 6))

    color = 0

    folder = 'process_data/{}_{}/'.format(dataset, fraction)
    length = 20

    ref_point = [0.3, 1.0] # TINYMNIST
    #
    # for c in range(num_classes):
    #     with open(os.path.join(folder, 'iter_0_label_{}/best_solution.pkl'.format(c)), 'rb') as f:
    #         best = pickle.load(f)
    #     best_index = 0
    #     fitness_list = []
    #     while best_index < len(best):
    #         ref_point[0] = max(ref_point[0], best[best_index].fitness[0])
    #         ref_point[1] = max(ref_point[1], best[best_index].fitness[1])
    #         best_index += 1

    ind = HV(ref_point=ref_point)
    average_hv_list = []
    write_data = {
        'iter': [],
        'hv': [],
    }
    iter = 0
    while iter < 100:
        pf = []
        hv_list = []
        for c in range(num_classes):
            with open(os.path.join(folder, 'iter_{}_label_{}/best_solution.pkl'.format(iter, c)), 'rb') as f:
                best_solution = pickle.load(f)
                # best_list = get_multi_best_solution(best_solution)
                # best = [best_solution[i] for i in best_list]
                best = best_solution
                if len(best) > length:
                    best = random.sample(best, length)
            best_index = 0
            fitness_list = []
            while best_index < len(best):

                class_index = np.arange(n_train)[dst_train.targets == c]


                # best = np.load('test_data/multi_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
                # best_index = np.load('test_data/iter_File_{}/multi_{}/best_index_{}.npy'.format(iter, dataset, fraction))[0]
                # best_index = best_index_dict[iter]
                # best_i.init(best[best_index])
                # print('best: ', best_i.fitness)
                fitness_list.append(best[best_index].fitness)
                best_index += 1

            fitness_array = np.array(fitness_list)
            print('label', c)
            print('length', length)

            hv_value = ind(fitness_array)
            print("HV", hv_value)
            hv_list.append(hv_value)
        print('iter: ', iter)
        average_hv = np.mean(np.array(hv_list))
        print('average hv: ', average_hv)
        average_hv_list.append(average_hv)
        write_data['iter'].append(iter)
        write_data['hv'].append(average_hv)
        iter += 5
    print(average_hv_list)
    df = pds.DataFrame(write_data)
    path = 'criterion_data/'
    os.makedirs(path, exist_ok=True)
    df.to_excel(os.path.join(path, 'hv_{}.xlsx'.format(fraction)), index=False)
    exit(0)


    # calculators = []
    # total_gene = None
    # gene = None
    # for c in range(10):
    #     features_matrix = torch.load(
    #         os.path.join(folder, 'features_matrix_{}.pth'.format(c))).cpu()
    #     confidence = torch.load(os.path.join(folder, 'importance_{}.pth'.format(c)))
    #     size = round(features_matrix.shape[0] * fraction)
    #     total_gene = features_matrix.shape[0]
    #     gene = size
    #     fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
    #                            InfoCalculator(features_matrix, confidence, size, device='cuda')]
    #     calculators.append(fitness_calculators)
    # for iter in best_index_dict.keys():
    #     best_index = 0
    #     length = 5
    #     while best_index < length:
    #         fitness_list = []
    #         for c in range(10):
    #             class_index = np.arange(n_train)[dst_train.targets == c]
    #
    #             with open(os.path.join(folder, 'iter_{}_label_{}/best_results.pkl'.format(iter, c)), 'rb') as f:
    #                 best = pickle.load(f)
    #                 length = len(best)
    #             # best = np.load('test_data/multi_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
    #             # best_index = np.load('test_data/iter_File_{}/multi_{}/best_index_{}.npy'.format(iter, dataset, fraction))[0]
    #             # best_index = best_index_dict[iter]
    #             fitness_calculators = calculators[c]
    #
    #             Individual.fitness_calculators = fitness_calculators
    #             best_i = Individual(total_gene, gene, 2)
    #             best_i.init(best[best_index])
    #             # print('best: ', best_i.fitness)
    #             fitness_list.append(best_i.fitness)
    #         fitness_array = np.array(fitness_list)
    #         average_fitness = np.mean(fitness_array, axis=0)
    #         print('iter: ', iter)
    #         print('index: ', best_index)
    #         print(average_fitness)
    #         best_index += 1


