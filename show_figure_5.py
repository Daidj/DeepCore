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
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator



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
    best_index_dict = {
        0: 0,
        10: 2,
        20: 0,
        30: 1,
        40: 0,
        50: 0,
        70: 2,
        90: 1
    }

    folder = 'process_data_70'
    length = 20
    IGDCalculators = []
    ## 计算IGD指标
    for c in range(10):
        with open(os.path.join(folder, 'iter_90_label_{}/best_solution'.format(c)), 'rb') as f:
            best = pickle.load(f)
            # length = min(len(best), length)
            # length = len(best)
            # if len(best) > length:
            #     best = random.sample(best, length)
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
        IGDCalculators.append(IGD(fitness_array))



    average_igd_list = []
    for iter in best_index_dict.keys():
        igd_list = []
        for c in range(10):
            with open(os.path.join(folder, 'iter_{}_label_{}/best_solution'.format(iter, c)), 'rb') as f:
                best = pickle.load(f)
                # length = min(len(best), length)
                # length = len(best)
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

            igd_value = IGDCalculators[c](fitness_array)
            print("IGD", igd_value)
            igd_list.append(igd_value)
        print('iter: ', iter)
        average_igd = np.mean(np.array(igd_list))
        print('average igd: ', average_igd)
        average_igd_list.append(average_igd)
    print(average_igd_list)

    write_data = {
        'iter': [],
        'igd': [],
    }

    df = pds.DataFrame(write_data)
    path = 'criterion_data/igd_{}'.format(fraction)
    os.makedirs(path, exist_ok=True)
    df.to_excel(path, index=False)

    exit(0)


    calculators = []
    for c in range(1):
        features_matrix = torch.load(
            os.path.join(folder, 'label_{}/features_matrix_{}_{}.pth'.format(c, fraction, dataset))).cpu()
        confidence = torch.load(os.path.join(folder, 'label_{}/importance_{}_{}.pth'.format(c, fraction, dataset)))
        size = round(features_matrix.shape[0] * fraction)

        fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
                               InfoCalculator(features_matrix, confidence, size, device='cuda')]
        calculators.append(fitness_calculators)
    for iter in best_index_dict.keys():
        best_index = 0
        length = 20
        while best_index < length:
            fitness_list = []
            for c in range(10):
                class_index = np.arange(n_train)[dst_train.targets == c]

                with open(os.path.join(folder, 'iter_{}_label_{}/best_solution.pkl'.format(iter, c)), 'rb') as f:
                    best = pickle.load(f)
                    length = len(best)
                # best = np.load('test_data/multi_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
                # best_index = np.load('test_data/iter_File_{}/multi_{}/best_index_{}.npy'.format(iter, dataset, fraction))[0]
                # best_index = best_index_dict[iter]
                fitness_calculators = calculators[c]

                Individual.fitness_calculators = fitness_calculators
                best_i = Individual(features_matrix.shape[0], size, 2)
                best_i.init(best[best_index])
                # print('best: ', best_i.fitness)
                fitness_list.append(best_i.fitness)
                continue
            fitness_array = np.array(fitness_list)
            average_fitness = np.mean(fitness_array, axis=0)
            print('iter: ', iter)
            print('index: ', best_index)
            print(average_fitness)
            best_index += 1

        # pareto_best_list = []
        # for i in range(len(best)):
        #     best_i = Individual(features_matrix.shape[0], size, 2)
        #     best_i.init(best[i].tolist())
        #     if i == best_index:
        #         best_individual_fitness = best_i.fitness
        #     print('best: ', best_i.fitness)
        #     pareto_best_list.append(best_i.fitness)
        # x = [point[0] for point in pareto_best_list]
        # y = [point[1] for point in pareto_best_list]
        # min_x = min(x)
        # max_x = max(x)
        # min_y = min(y)
        # max_y = max(y)
        # plt.scatter(x, y, s=5, c=colors[color])
        # color += 1
        #
        # best = np.load('test_data/ldea_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
        # # best_index = np.load('test_data/ldea_{}/best_index_{}.npy'.format(dataset, fraction))[0]
        # best_index = 0
        #
        # pareto_best_list = []
        # for i in range(len(best)):
        #     best_i = Individual(features_matrix.shape[0], size, 2)
        #     best_i.init(best[i].tolist())
        #     if i == best_index:
        #         best_individual_fitness = best_i.fitness
        #     print('best: ', best_i.fitness)
        #     pareto_best_list.append(best_i.fitness)
        # x = [point[0] for point in pareto_best_list]
        # y = [point[1] for point in pareto_best_list]
        # # min_x = min(min_x, min(x))
        # # max_x = max(max_x, max(x))
        # # min_y = min(min_y, min(y))
        # # max_y = max(max_y, max(y))
        # min_x = min(x)
        # max_x = max(x)
        # min_y = min(y)
        # max_y = max(y)
        # plt.scatter(x, y, s=5, c=colors[color])
        # for i, j in zip(x, y):
        #     plt.annotate(f'pareto:({i:.5f},{j:.5f})', (i, j), color=colors[color])
        # color += 1
        #
        # diff_x = [best_individual_fitness[0]]
        # diff_y = [best_individual_fitness[1]]
        # min_x = min(min_x, min(diff_x))
        # max_x = max(max_x, max(diff_x))
        # min_y = min(min_y, min(diff_y))
        # max_y = max(max_y, max(diff_y))
        # plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        # for i, j in zip(diff_x, diff_y):
        #     plt.annotate(f'best:({i:.5f},{j:.5f})', (i, j), color=colors[color])
        # color += 1
        #
        # result = np.argsort(confidence)[:round(len(class_index) * fraction)]
        # leastconfidence_individual = Individual(features_matrix.shape[0], size, 2)
        # leastconfidence_individual.init(result.tolist())
        # print('leastconfidence: ', leastconfidence_individual.fitness)
        # leastconfidence_best = leastconfidence_individual.fitness
        # diff_x = [leastconfidence_best[0]]
        # diff_y = [leastconfidence_best[1]]
        # min_x = min(min_x, min(diff_x))
        # max_x = max(max_x, max(diff_x))
        # min_y = min(min_y, min(diff_y))
        # max_y = max(max_y, max(diff_y))
        # plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        # for i, j in zip(diff_x, diff_y):
        #     plt.annotate(f'leastconfidence({i:.5f},{j:.5f})', (i, j), color=colors[color])
        # color += 1
        #
        # res_greedy = torch.from_numpy(
        #     k_center_greedy(matrix=features_matrix, budget=size, metric=euclidean_dist,
        #                     device='cuda'))
        # individual = Individual(features_matrix.shape[0], size, 2)
        # individual.init(res_greedy.tolist())
        # print('kcenter: ', individual.fitness)
        # kcenter_best = individual.fitness
        # diff_x = [kcenter_best[0]]
        # diff_y = [kcenter_best[1]]
        # min_x = min(min_x, min(diff_x))
        # max_x = max(max_x, max(diff_x))
        # min_y = min(min_y, min(diff_y))
        # max_y = max(max_y, max(diff_y))
        # plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        # for i, j in zip(diff_x, diff_y):
        #     plt.annotate(f'kcenter({i:.5f},{j:.5f})', (i, j), color=colors[color])
        # color += 1
        #
        # result = np.random.choice(range(features_matrix.shape[0]), size=size, replace=False)
        # individual = Individual(features_matrix.shape[0], size, 2)
        # individual.init(result.tolist())
        # print('random: ', individual.fitness)
        # random_best = individual.fitness
        # diff_x = [random_best[0]]
        # diff_y = [random_best[1]]
        # min_x = min(min_x, min(diff_x))
        # max_x = max(max_x, max(diff_x))
        # min_y = min(min_y, min(diff_y))
        # max_y = max(max_y, max(diff_y))
        # plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        # for i, j in zip(diff_x, diff_y):
        #     plt.annotate(f'random({i:.5f},{j:.5f})', (i, j), color=colors[color])
        # color += 1

        # plt.xlim(min_x - 0.2 * (max_x - min_x), max_x + 0.2 * (max_x - min_x))
        # plt.ylim(min_y - 0.2 * (max_y - min_y), max_y + 0.2 * (max_y - min_y))
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('All result fitness')
        # os.makedirs(folder_name, exist_ok=True)
        # file_path = os.path.join(folder_name, '{}.png'.format(title))
        # plt.savefig(file_path)
        # plt.show()
