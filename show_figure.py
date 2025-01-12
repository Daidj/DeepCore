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
    height, width, _ = img.shape

    plt.axis('off')
    plt.text(-width * 0.05, -height * 0.01, '{}:{}'.format(text, index),
             ha='center', va='bottom', fontsize=14)
    folder = os.path.join(save_folder, '{}'.format(text))
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, '{}.png'.format(index)))
    # plt.show()
    plt.close()


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
    datas = []
    best = []
    class_num = [0, 2, 7, 9]
    for c in range(13, 14):

        folder = 'test_data/multi_{}'.format(c)
        best_file_path = os.path.join(folder, 'best_{}_multi_{}.npy'.format(fraction, dataset))
        features_matrix = torch.load(
            os.path.join(folder, 'features_matrix_{}_{}.pth'.format(fraction, dataset))).cpu().numpy()
        confidence = torch.load(os.path.join(folder, 'importance_{}_{}.pth'.format(fraction, dataset)))
        best_result = np.load(best_file_path)
        best_index = np.load('test_data/multi_{}/best_index_{}.npy'.format(dataset, fraction))
        best.append(best_result[best_index[0]])
        # best.append(np.random.choice(np.arange(features_matrix.shape[0]), size=len(best_result[0]), replace=False))

        # datas.append(features_matrix)
        datas.append(features_matrix[best[-1]])
        # datas = []
        # range_1 = [0, 9]
        # for i in range(1):
        #     features_matrix = torch.load('test_data/features_matrix_{}.pt'.format(i)).cpu()
        #     data1 = features_matrix.numpy()
        #     datas.append(data1)

    data = np.concatenate(datas, axis=0)
    # pca = PCA(n_components=2)
    # X_transformed = pca.fit_transform(data)
    # print("\n主成分方差比例:")
    # print(pca.explained_variance_ratio_)
    tsne = TSNE(n_components=2)
    X_transformed = tsne.fit_transform(data)
    # arr = np.arange(5000)
    #
    # # 从这个数组中随机挑选n个元素，允许重复
    # indice = np.random.choice(arr, size=500, replace=False)
    # X_transformed = X_transformed[indice]
    print(X_transformed.shape)
    num = int(X_transformed.shape[0]/len(best))
    # plt.xlim(-40, 55)
    # plt.ylim(-45, 50)
    plt.xlim(round(min(X_transformed[:, 0]))-2, round(max(X_transformed[:, 0]))+2)
    plt.ylim(round(min(X_transformed[:, 1]))-2, round(max(X_transformed[:, 1]))+2)
    for i in range(len(best)):
        center = np.mean(X_transformed[i*num:(i+1)*num], axis=0)
        plt.scatter(X_transformed[i*num:(i+1)*num][:, 0], X_transformed[i*num:(i+1)*num][:, 1],
                    c=colors[color],
                    s=5)
        color += 1
        # plt.scatter([center[0]], [center[1]],
        #             c=colors[color],
        #             s=10)
        # color += 1
        # plt.scatter(X_transformed[i*num:(i+1)*num][best, 0], X_transformed[i*num:(i+1)*num][best, 1],
        #             c=colors[color],
        #             s=5)
        # color += 1

        # 添加标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of n*2 NumPy Array')

    # 显示图形
    plt.show()

        # kde = gaussian_kde(X_transformed.T)
        # min_x = X_transformed.T[0].min()
        # max_x = X_transformed.T[0].max()
        # min_y = X_transformed.T[1].min()
        # max_y = X_transformed.T[1].max()
        # x, y = np.mgrid[min_x:max_x:100j, min_y:max_y:100j]
        # positions = np.vstack([x.ravel(), y.ravel()])
        # f = np.reshape(kde(positions).T, x.shape)
        # plt.figure(figsize=(8, 6))
        # plt.contourf(x, y, f, levels=30, cmap='viridis')
        # plt.colorbar()
        # plt.title('Gaussian KDE of 2D Data')
        # plt.show()

    print('hello world!')
