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

from deepcore.methods import MMDCalculator, k_center_greedy, euclidean_dist_for_batch, euclidean_dist
from deepcore.methods.micro import micro
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
    # front_tensor = torch.tensor([[3.0, 0.0], [1.5, 0.0], [2.0, 0.0]])
    # best_point = torch.min(front_tensor, dim=0).values
    # worst_point = torch.max(front_tensor, dim=0).values
    # scores = (front_tensor - best_point) / (worst_point - best_point+1e-8)
    # fraction = torch.tensor([0.5, 0.5])
    # fraction_matrix = fraction.unsqueeze(0).repeat(scores.size(0), 1)
    # scores = torch.sum(scores * fraction_matrix, dim=1)
    # best = torch.argmin(scores)
    # print(best)

    # rate = 0.1
    # sum = 0.0
    # i = 0
    # while sum < 1.0:
    #     sum += rate
    #     rate = rate*0.9
    #     i += 1
    # print(i)
    # exit(0)

    # import torch
    # list1 = [[2, 8, 5], [1, 6, 9], [2, 2, 2]]
    # tensor = torch.tensor(list1)
    #
    # print(tensor)
    # print(torch.exp(-tensor))

    # scatter_features_2()
    # exit(0)
    # torch.save(torch.from_numpy(frequency), 'test_data/frequency_{}.pt'.format(c))
    for c in range(3):
        # data1 = torch.load('test_data/importance_{}.pt'.format(c)).cpu().numpy()
        # features_matrix = torch.load('test_data/features_matrix_{}.pt'.format(c)).cpu().numpy()
        # pca = PCA(n_components=1)
        # data2 = pca.fit_transform(features_matrix).ravel()
        # print(data1.shape)
        # print(data2.shape)
        # label1 = "P - importance"
        # label2 = "Q - features"
        # kl_div = run_test(data1, data2, label1, label2)
        # print('kl: ', kl_div)
        # data1 = torch.load('test_data/importance_{}.pt'.format(c)).cpu().numpy()
        print(c)
        dataset = 'CIFAR100'
        # features_matrix = torch.load('test_data/{}/features_matrix_{}.pth'.format(c, dataset)).cpu().numpy()
        # confidence = torch.load('test_data/{}/importance_{}.pth'.format(c, dataset))
        features_matrix = torch.load('test_data/features_matrix_{}.pt'.format(c)).cpu()
        confidence = torch.load('test_data/importance_{}.pt'.format(c))
        # center = torch.mean(features_matrix, dim=0)
        # center = center.reshape(1, -1)
        center = features_matrix[torch.argmin(confidence)]
        center = center.reshape(1, -1)
        dis = euclidean_dist(center, features_matrix)


        total_size = features_matrix.shape[0]
        num = round(0.1*total_size)
        selected = torch.from_numpy(self_adaptation_search(features_matrix, confidence, num, euclidean_dist, 'cuda')[0])
        # distance = calculator.mmd_for_data_set(selected)
        print('Micro:', len(selected))
        continue

        calculator = MMD(features_matrix, 'cuda')
        # selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), 1))
        # distance = calculator.mmd_for_data_set(selected)
        # print('0-1:', distance)
        selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        distance = calculator.mmd_for_data_set(selected)
        print('0-100:', distance)
        selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        distance = calculator.mmd_for_data_set(selected)
        print('0-500:', distance)
        selected = torch.from_numpy(two_stage_search(features_matrix, confidence, num, euclidean_dist, 'cuda')[0])
        distance = calculator.mmd_for_data_set(selected)
        print('Micro:', distance)
        continue

        selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        distance = calculator.mmd_for_data_set(selected)
        print('0-1000:', distance)
        selected = torch.tensor(random.sample(list(torch.arange(total_size).numpy()), num))
        distance = calculator.mmd_for_data_set(selected)
        print('0-2000:', distance)

        selected = torch.from_numpy(k_center_uncertainty_greedy(features_matrix, confidence, num, euclidean_dist, 'cuda'))
        distance = calculator.mmd_for_data_set(selected)
        print('kcenter-uncertainty:', distance)
        selected = torch.tensor(calculator.get_min_distance_index(num))
        distance = calculator.mmd_for_data_set(selected)
        print('best:', distance)

        selected = torch.from_numpy(
            k_center_greedy(matrix=features_matrix, budget=num, metric=euclidean_dist,
                            device='cuda'))
        distance = calculator.mmd_for_data_set(selected)
        print('kcenter:', distance)
        # selected = torch.tensor(calculator.get_min_distance_index_2(500))
        # distance = calculator.mmd_for_data_set(selected)
        # print('best2:', distance)

        unselected = set(torch.arange(total_size).numpy())
        unselected = unselected - set(selected)
        # calculator = MMDCalculator(features_matrix, 500, 'cuda')
        calculator.get_selected_scores(selected, unselected)

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
