# from deepcore.methods.moea_d_ldea import SubProblems
import math
import os
import random

import numpy
import matplotlib.pyplot as plt
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


def scatter_features():

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.figure(figsize=(8, 6))
    datas = []
    range_1 = [0, 9]
    for i in range(1):
        features_matrix = torch.load('test_data/features_matrix_{}.pt'.format(i)).cpu()
        data1 = features_matrix.numpy()
        datas.append(data1)

    data = np.concatenate(datas, axis=0)
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(data)
    print("\n主成分方差比例:")
    print(pca.explained_variance_ratio_)
    # arr = np.arange(5000)
    #
    # # 从这个数组中随机挑选n个元素，允许重复
    # indice = np.random.choice(arr, size=500, replace=False)
    # X_transformed = X_transformed[indice]
    print(X_transformed.shape)

    for c in range(1):

        plt.scatter(X_transformed[c * 5000:(c + 1) * 5000, 0], X_transformed[c * 5000:(c + 1) * 5000, 1], c=colors[c],
                    s=1)
    max_indices = np.argmax(data, axis=0)
    print(len(set(max_indices)))
    for i in max_indices:
        plt.scatter(X_transformed[i, 0], X_transformed[i, 1], c=colors[5], s=5)

    # 添加标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of n*2 NumPy Array')

    # 显示图形
    plt.show()

def scatter_features_2():

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.figure(figsize=(8, 6))
    datas = []
    range_1 = [0, 9]
    for i in range(1):
        features_matrix = torch.load('test_data/features_matrix_{}.pt'.format(i)).cpu()
        data1 = features_matrix.numpy()
        datas.append(data1)

    data = np.concatenate(datas, axis=0)
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(data)
    print("\n主成分方差比例:")
    print(pca.explained_variance_ratio_)
    # arr = np.arange(5000)
    #
    # # 从这个数组中随机挑选n个元素，允许重复
    # indice = np.random.choice(arr, size=500, replace=False)
    # X_transformed = X_transformed[indice]
    print(X_transformed.shape)

    for c in range(1):

        plt.scatter(X_transformed[c * 5000:(c + 1) * 5000, 0], X_transformed[c * 5000:(c + 1) * 5000, 1], c=colors[c],
                    s=1)

    # 添加标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of n*2 NumPy Array')

    # 显示图形
    plt.show()

    kde = gaussian_kde(X_transformed.T)
    min_x = X_transformed.T[0].min()
    max_x = X_transformed.T[0].max()
    min_y = X_transformed.T[1].min()
    max_y = X_transformed.T[1].max()
    x, y = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    f = np.reshape(kde(positions).T, x.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, f, levels=30, cmap='viridis')
    plt.colorbar()
    plt.title('Gaussian KDE of 2D Data')
    plt.show()


if __name__ == '__main__':
    data_path = 'data'

    dataset = 'CIFAR100'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset](data_path)
    n_train = len(dst_train)
    best = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fraction = 0.7
    plt.figure(figsize=(8, 6))

    color = 0

    for c in range(5, 6):

        print(c)
        class_index = np.arange(n_train)[dst_train.targets == c]

        # features_matrix = torch.load('test_data/{}/features_matrix_{}.pth'.format(c, dataset)).cpu().numpy()
        # confidence = torch.load('test_data/{}/importance_{}.pth'.format(c, dataset))
        features_matrix = torch.load('test_data/multi_{}/features_matrix_{}_{}.pth'.format(c, fraction, dataset)).cpu()
        confidence = torch.load('test_data/multi_{}/importance_{}_{}.pth'.format(c, fraction, dataset))
        best = np.load('test_data/multi_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
        best_index = np.load('test_data/multi_{}/best_index_{}.npy'.format(dataset, fraction))[0]

        size = round(features_matrix.shape[0] * fraction)

        fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
                               InfoCalculator(features_matrix, confidence, size, device='cuda')]
        Individual.fitness_calculators = fitness_calculators

        pareto_best_list = []
        for i in range(len(best)):
            best_i = Individual(features_matrix.shape[0], size, 2)
            best_i.init(best[i].tolist())
            if i == best_index:
                best_individual_fitness = best_i.fitness
            print('best: ', best_i.fitness)
            pareto_best_list.append(best_i.fitness)
        x = [point[0] for point in pareto_best_list]
        y = [point[1] for point in pareto_best_list]
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        plt.scatter(x, y, s=5, c=colors[color])
        color += 1

        best = np.load('test_data/ldea_{}/best_{}_multi_{}.npy'.format(c, fraction, dataset))
        # best_index = np.load('test_data/ldea_{}/best_index_{}.npy'.format(dataset, fraction))[0]
        best_index = 0

        pareto_best_list = []
        for i in range(len(best)):
            best_i = Individual(features_matrix.shape[0], size, 2)
            best_i.init(best[i].tolist())
            if i == best_index:
                best_individual_fitness = best_i.fitness
            print('best: ', best_i.fitness)
            pareto_best_list.append(best_i.fitness)
        x = [point[0] for point in pareto_best_list]
        y = [point[1] for point in pareto_best_list]
        # min_x = min(min_x, min(x))
        # max_x = max(max_x, max(x))
        # min_y = min(min_y, min(y))
        # max_y = max(max_y, max(y))
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        plt.scatter(x, y, s=5, c=colors[color])
        for i, j in zip(x, y):
            plt.annotate(f'pareto:({i:.5f},{j:.5f})', (i, j), color=colors[color])
        color += 1

        diff_x = [best_individual_fitness[0]]
        diff_y = [best_individual_fitness[1]]
        min_x = min(min_x, min(diff_x))
        max_x = max(max_x, max(diff_x))
        min_y = min(min_y, min(diff_y))
        max_y = max(max_y, max(diff_y))
        plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'best:({i:.5f},{j:.5f})', (i, j), color=colors[color])
        color += 1

        result = np.argsort(confidence)[:round(len(class_index) * fraction)]
        leastconfidence_individual = Individual(features_matrix.shape[0], size, 2)
        leastconfidence_individual.init(result.tolist())
        print('leastconfidence: ', leastconfidence_individual.fitness)
        leastconfidence_best = leastconfidence_individual.fitness
        diff_x = [leastconfidence_best[0]]
        diff_y = [leastconfidence_best[1]]
        min_x = min(min_x, min(diff_x))
        max_x = max(max_x, max(diff_x))
        min_y = min(min_y, min(diff_y))
        max_y = max(max_y, max(diff_y))
        plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'leastconfidence({i:.5f},{j:.5f})', (i, j), color=colors[color])
        color += 1

        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=features_matrix, budget=size, metric=euclidean_dist,
                            device='cuda'))
        individual = Individual(features_matrix.shape[0], size, 2)
        individual.init(res_greedy.tolist())
        print('kcenter: ', individual.fitness)
        kcenter_best = individual.fitness
        diff_x = [kcenter_best[0]]
        diff_y = [kcenter_best[1]]
        min_x = min(min_x, min(diff_x))
        max_x = max(max_x, max(diff_x))
        min_y = min(min_y, min(diff_y))
        max_y = max(max_y, max(diff_y))
        plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'kcenter({i:.5f},{j:.5f})', (i, j), color=colors[color])
        color += 1

        result = np.random.choice(range(features_matrix.shape[0]), size=size, replace=False)
        individual = Individual(features_matrix.shape[0], size, 2)
        individual.init(result.tolist())
        print('random: ', individual.fitness)
        random_best = individual.fitness
        diff_x = [random_best[0]]
        diff_y = [random_best[1]]
        min_x = min(min_x, min(diff_x))
        max_x = max(max_x, max(diff_x))
        min_y = min(min_y, min(diff_y))
        max_y = max(max_y, max(diff_y))
        plt.scatter(diff_x, diff_y, s=5, c=colors[color])
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'random({i:.5f},{j:.5f})', (i, j), color=colors[color])
        color += 1

        # test_data_folder = 'test_data/origin_moea{}'.format(c)
        # solution_num = 5
        # step_rate = 0.1
        # iter = 100
        # population_num = 50




        # for i, j in zip(x, y):
        #     plt.annotate(f'({i:.2f},{j:.2f})', (i, j))

        # if important_points != None:
        #     important_x = [point[0] for point in important_points]
        #     important_y = [point[1] for point in important_points]
        #     min_x = min(min_x, min(important_x))
        #     max_x = max(max_x, max(important_x))
        #     min_y = min(min_y, min(important_y))
        #     max_y = max(max_y, max(important_y))
        #     plt.scatter(important_x, important_y, s=5, c='g')
        #     for i, j in zip(important_x, important_y):
        #         plt.annotate(f'({i:.5f},{j:.5f})', (i, j), color='green')

        plt.xlim(min_x - 0.2 * (max_x - min_x), max_x + 0.2 * (max_x - min_x))
        plt.ylim(min_y - 0.2 * (max_y - min_y), max_y + 0.2 * (max_y - min_y))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('All result fitness')
        # os.makedirs(folder_name, exist_ok=True)
        # file_path = os.path.join(folder_name, '{}.png'.format(title))
        # plt.savefig(file_path)
        plt.show()


        # total_size = features_matrix.shape[0]
        # num = round(0.1*total_size)
        # selected = torch.from_numpy(self_adaptation_search(features_matrix, confidence, num, euclidean_dist, 'cuda')[0])
        # # distance = calculator.mmd_for_data_set(selected)
        # print('Micro:', len(selected))
        # continue
        #
        # calculator = MMD(features_matrix, 'cuda')
        # # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), 1))
        # # distance = calculator.mmd_for_data_set(selected)
        # # print('0-1:', distance)
        # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        # distance = calculator.mmd_for_data_set(selected)
        # print('0-100:', distance)
        # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        # distance = calculator.mmd_for_data_set(selected)
        # print('0-500:', distance)
        # selected = torch.from_numpy(two_stage_search(features_matrix, confidence, num, euclidean_dist, 'cuda')[0])
        # distance = calculator.mmd_for_data_set(selected)
        # print('Micro:', distance)
        # continue
        #
        # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        # distance = calculator.mmd_for_data_set(selected)
        # print('0-1000:', distance)
        # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        # distance = calculator.mmd_for_data_set(selected)
        # print('0-2000:', distance)
        #
        # selected = torch.from_numpy(k_center_uncertainty_greedy(features_matrix, confidence, num, euclidean_dist, 'cuda'))
        # distance = calculator.mmd_for_data_set(selected)
        # print('kcenter-uncertainty:', distance)
        # selected = torch.tensor(calculator.get_min_distance_index(num))
        # distance = calculator.mmd_for_data_set(selected)
        # print('best:', distance)
        #
        # selected = torch.from_numpy(
        #     k_center_greedy(matrix=features_matrix, budget=num, metric=euclidean_dist,
        #                     device='cuda'))
        # distance = calculator.mmd_for_data_set(selected)
        # print('kcenter:', distance)
        # # selected = torch.tensor(calculator.get_min_distance_index_2(500))
        # # distance = calculator.mmd_for_data_set(selected)
        # # print('best2:', distance)
        #
        # unselected = set(torch.arange(total_size).numpy())
        # unselected = unselected - set(selected)
        # # calculator = MMDCalculator(features_matrix, 500, 'cuda')
        # calculator.get_selected_scores(selected, unselected)

        # 创建图形对象
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor='#f0f0f0')
        #
        # # 绘制第一组数据的频率分布图
        # ax[0].hist(data1, bins=20, density=True, edgecolor='black', facecolor='#1f77b4')
        # ax[0].set_title('Frequency Distribution - Data 1')
        # ax[0].set_xlabel('Value')
        # ax[0].set_ylabel('Frequency')
        #
        # # 绘制第二组数据的频率分布图
        # ax[1].hist(data2, bins=20, density=True, edgecolor='black', facecolor='#ff7f0e')
        # ax[1].set_title('Frequency Distribution - Data 2')
        # ax[1].set_xlabel('Value')
        # ax[1].set_ylabel('Frequency')
        #
        # # 显示图像
        # plt.show()



# import numpy as np
    # from scipy.stats import gaussian_kde
    # import matplotlib.pyplot as plt
    #
    # # 生成一些随机的二维数据
    # X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=500)
    #
    # # 使用 gaussian_kde 估计密度
    # kde = gaussian_kde(X.T)
    #
    # # 创建一个网格来评估密度
    # x, y = np.mgrid[-4:4:.05, -4:4:.05]
    # positions = np.vstack([x.ravel(), y.ravel()])
    # f = np.reshape(kde(positions).T, x.shape)
    #
    # # 绘制密度图
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.contourf(x, y, f, levels=30, cmap='Blues')
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_title('Density Estimation using Gaussian KDE')
    # plt.show()
