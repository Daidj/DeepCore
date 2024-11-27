import itertools
import math
import random
import time
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

# def euclidean_dist(x, y):
#     x = x.float()
#     y = y.float()
#     m, n = x.size(0), y.size(0)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     dist.addmm_(x, y.t(), beta=1, alpha=-2)
#     # dist.addmm_(1, -2, x, y.t())
#     dist = dist.clamp(min=1e-12).sqrt()
#     # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
#     return dist
#
# def euclidean_dist_for_batch(x, y, batch=2048, metric=euclidean_dist):
#
#     # dist: m * n, dist[i][j] 表示x中的第i个样本和y中的第j个样本的距离
#     m, n = x.size(0), y.size(0)
#     distance = torch.empty(m, n, dtype=x.dtype).to(device='cpu')
#     i = 0
#     while True:
#         row_start = i * batch
#         row_end = min((i + 1) * batch, m)
#         distance[row_start:row_end] = metric(x[row_start:row_end], y).cpu()
#         if row_end == m:
#             break
#         i = i + 1
#     # print(distance)
#     return distance
from deepcore.methods.methods_utils import euclidean_dist_for_batch, euclidean_dist


class MMD:
    def __init__(self, matrix, device='cuda'):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        self.features_matrix = matrix.to(device=device)
        self.dataset_size = matrix.shape[0]
        self.device = device
        self.distance = euclidean_dist_for_batch(self.features_matrix, self.features_matrix).cpu()
        self.min_distance = 1e-6
        max_dis, _ = torch.max(self.distance, dim=1)
        self.max_distance = torch.max(max_dis).cpu().item() + 1e-6
        self.kernel_XX = None

    def get_min_distance_index_2(self, selected_size):

        result = set()
        unselected = set(range(self.dataset_size))
        for i in range(selected_size):
            if len(result) == 0:
                selected = random.choice(list(unselected))
            else:
                selected_tensor = torch.tensor(list(result))
                unselected_tensor = torch.tensor(list(unselected))
                distance = self.distance[unselected_tensor, :]
                selected_distance = self.distance[unselected_tensor][:, selected_tensor]
                # print(selected_distance.shape)
                samples_scores = torch.mean(distance, dim=1) - torch.mean(selected_distance, dim=1)
                selected = list(unselected)[torch.argmin(samples_scores).cpu().item()]
            result.add(selected)
            unselected.remove(selected)
        return list(result)

    def get_min_distance_index(self, selected_size, step_rate=None):

        selected = set()
        unselected = set(range(self.dataset_size))
        if step_rate is None:
            for loop in range(selected_size):
                if len(selected) == 0:
                    element = random.choice(list(unselected))
                else:
                    # selected_tensor = torch.tensor(list(result))
                    # unselected_tensor = torch.tensor(list(unselected))
                    # distance = self.distance[unselected_tensor, :]
                    # selected_distance = self.distance[unselected_tensor][:, selected_tensor]
                    # print(selected_distance.shape)
                    samples_scores = self.get_unselected_scores(selected, unselected)
                    element = list(unselected)[torch.argmin(samples_scores).cpu().item()]
                selected.add(element)
                unselected.remove(element)
        else:
            step = max(1, round(step_rate*selected_size))
            while selected_size > 0:
                selected_num = min(selected_size, step)
                selected_num = max(1, selected_num)
                if len(selected) == 0:
                    # selected_num = max(5, selected_num)
                    selected = set(random.sample(list(unselected), selected_num))
                    unselected.difference_update(selected)
                else:
                    mmd_scores = self.get_unselected_scores(selected, unselected)
                    _, indices = torch.topk(mmd_scores, k=selected_num, largest=False)
                    l = list(unselected)
                    new_elements = set([l[i] for i in indices])
                    selected.update(new_elements)
                    unselected.difference_update(selected)
                selected_size -= selected_num
                step = math.floor(step * 0.9)
        return list(selected)

    def get_unselected_scores(self, selected, unselected):
        selected_tensor = torch.tensor(list(selected))
        unselected_tensor = torch.tensor(list(unselected))
        # distance = self.distance[unselected_tensor, :]
        # selected_distance = self.distance[unselected_tensor][:, selected_tensor]
        # print(selected_distance.shape)
        # samples_scores = (torch.mean(distance, dim=1) - torch.mean(selected_distance, dim=1)).cpu()
        # samples_scores = (samples_scores - self.min_distance)/(self.max_distance-self.min_distance)
        # samples_scores = (samples_scores - samples_scores.min())/(samples_scores.max()-samples_scores.min())
        if self.kernel_XX is None:
            self.mmd_for_data_set(selected_tensor)
        XY = self.kernel_XX[unselected_tensor]
        YY = self.kernel_XX[unselected_tensor][:, selected_tensor]
        n = self.dataset_size
        m = len(selected)+1
        XY = 2 * torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target
        YY = 2 * torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

        samples_scores = (XY+YY).squeeze()
        return samples_scores.cpu()

    def get_selected_scores(self, selected, unselected):
        selected_tensor = torch.tensor(list(selected))
        unselected_tensor = torch.tensor(list(unselected))
        XY = self.kernel_XX[selected_tensor]
        YY = self.kernel_XX[selected_tensor][:, selected_tensor]
        n = self.dataset_size
        m = len(selected)
        XY = 2 * torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target
        YY = 2 * torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

        samples_scores = (XY+YY).squeeze()
        return samples_scores.cpu()

    def get_distance(self, target_index):
        # target_index: 目标数据集的索引集，tensor
        # print("get distance:")
        length = self.dataset_size + target_index.size()[0]
        distance_for_two_set = torch.empty(length, length, dtype=torch.float32)
        distance_for_two_set[:self.dataset_size, :self.dataset_size] = self.distance
        distance_for_two_set[:self.dataset_size, self.dataset_size:] = self.distance[:, target_index]
        distance_for_two_set[self.dataset_size:, :self.dataset_size] = self.distance[target_index, :]
        distance_for_two_set[self.dataset_size:, self.dataset_size:] = self.distance[target_index][:, target_index]
        return distance_for_two_set

    def guassian_kernel(self, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算Gram核矩阵
        target: sample_size 的数据索引
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (dataset_size + sample_size) * (dataset_size + sample_size)的矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(self.dataset_size) + int(target.size()[0])
        # L2_distance = L1_distance(total0, total1) # 计算高斯核中的|x-y|
        L2_distance = self.get_distance(target)

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # 将多个核合并在一起

    def mmd(self, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        # start_time = time.time()

        n = self.dataset_size
        m = int(target.size()[0])

        kernels = self.guassian_kernel(target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        self.kernel_XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

        loss = (XX + XY).sum() + (YX + YY).sum()
        # end_time = time.time()
        # print("mmd time: ", end_time - start_time)
        return loss

    def mmd_for_data_set(self, target_index):
        return self.mmd(target_index).cpu().item()


class MMD_old:
    def __init__(self, matrix, device='cuda'):
        if type() == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        self.feature_matrix = matrix.to(device=device)
        self.dataset_size = matrix.shape[0]
        self.device = device
        self.distance_calculated = False

    def get_min_distance_index_in_range(self, size, subset):
        # 在subset索引集中寻找最小的mmd距离
        if not self.distance_calculated:
            self.calculate_L_distance()
        socres_array = self.distance
        origin_samples_scores = socres_array.sum(axis=0) / self.dataset_size
        samples_scores = origin_samples_scores[subset]
        result = []
        for i in range(size):
            min_index = subset[torch.argmin(samples_scores)]

            result.append(min_index.cpu().item())

            changed_score = self.distance[result].sum(axis=0) / (i + 1)

            samples_scores = origin_samples_scores - changed_score
            max_value = torch.max(samples_scores)
            samples_scores[result] = max_value
            samples_scores = samples_scores[subset]
        return result

    def get_min_distance_index(self, size):
        if not self.distance_calculated:
            self.calculate_L_distance()
        socres_array = self.distance
        origin_samples_scores = socres_array.sum(axis=0) / self.dataset_size
        samples_scores = origin_samples_scores
        result = []
        for i in range(size):
            min_index = torch.argmin(samples_scores)

            result.append(min_index.cpu().item())

            changed_score = self.distance[result].sum(axis=0) / (i + 1)

            samples_scores = origin_samples_scores - changed_score
            max_value = torch.max(samples_scores)
            samples_scores[result] = max_value
        return result

    def get_min_distance_index_with_random(self, size, random_num=1):
        if not self.distance_calculated:
            self.calculate_L_distance()
        socres_array = self.distance
        origin_samples_scores = socres_array.sum(axis=0) / self.dataset_size
        samples_scores = origin_samples_scores
        result = []
        for i in range(size):
            # 对张量进行排序并取出前n个数
            top_values, top_indices = torch.topk(samples_scores, k=random_num, largest=False)

            top_value = top_values.max() + top_values.max() / 1000 if top_values.max() != 0 else top_values.max() + 0.001
            total_scores = (top_value - top_values).sum()
            # 构建轮盘
            roulette_wheel = (top_value - top_values) / total_scores
            # 生成随机数，模拟轮盘赌选择
            random_number = random.random()
            # 轮盘赌选择
            cumulative_prob = 0
            selected_index = top_indices[0]
            for i, prob in enumerate(roulette_wheel):
                cumulative_prob += prob
                if random_number < cumulative_prob:
                    selected_index = top_indices[i]
                    break
            # min_index = torch.argmin(samples_scores)

            result.append(selected_index.cpu().item())
            changed_score = self.distance[result].sum(axis=0) / (i + 1)

            samples_scores = origin_samples_scores - changed_score
            max_value = torch.max(samples_scores)
            samples_scores[result] = max_value
        return result

    def calculate_L_distance(self):

        dataset_size = self.feature_matrix.shape[0]
        dataset = self.feature_matrix.reshape(dataset_size, -1).type(torch.int32).to(device='cuda')

        if not self.distance_calculated:
            dataset_copy = dataset.unsqueeze(0).expand(int(dataset.size(0)), int(dataset.size(0)), int(dataset.size(1)))
            samples_copy = dataset.unsqueeze(1).expand(int(dataset.size(0)), int(dataset.size(0)), int(dataset.size(1)))
            self.distance = self.L1_distance_for_loop(dataset_copy, samples_copy, 256)
            self.distance_calculated = True
        return self.distance

    def get_distance(self, target_index):
        # target_index: 目标数据集的索引集，tensor
        # print("get distance:")
        length = self.dataset_size + target_index.size()[0]
        distance_for_two_set = torch.empty(length, length, dtype=torch.int32).to(device='cuda')
        if not self.distance_calculated:
            self.calculate_L_distance()
        for i in range(length):
            if i < self.dataset_size:
                distance_for_two_set[i][0:self.dataset_size] = self.distance[i]
                distance_for_two_set[i][self.dataset_size:length] = self.distance[i][target_index]
            else:
                distance_for_two_set[i] = distance_for_two_set[target_index[i - self.dataset_size]]
        # print("distance_for_two_set:")
        # print(distance_for_two_set)
        return distance_for_two_set

    def L1_distance(self, x, y):
        start_time = time.time()
        # print(x.shape)
        # print(y.shape)
        distance = ((x - y).abs()).sum(2)
        end_time = time.time()
        print(end_time - start_time)
        return distance

    def L1_distance_for_loop(self, x, y, branch):
        start_time = time.time()
        len = x.shape[0]
        distance = torch.empty(len, len, dtype=x.dtype).to(device='cuda')
        i = 0
        while True:
            row_start = i * branch
            if i * branch >= len:
                row_end = len
            else:
                row_end = (i + 1) * branch
            j = 0
            while True:
                col_start = j * branch
                if j * branch >= len:
                    col_end = len
                else:
                    col_end = (j + 1) * branch

                # print(x[row_start:row_end, col_start:col_end].shape)
                # print(y[row_start:row_end, col_start:col_end].shape)
                distance[row_start:row_end, col_start:col_end] = (
                        x[row_start:row_end, col_start:col_end] - y[row_start:row_end,
                                                                  col_start:col_end]).abs().sum(2)
                if col_end == len:
                    break
                j = j + 1
            if row_end == len:
                break
            i = i + 1
        # print(distance)
        end_time = time.time()
        print(end_time - start_time)
        return distance

    def guassian_kernel(self, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算Gram核矩阵
        target: sample_size 的数据索引
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (dataset_size + sample_size) * (dataset_size + sample_size)的矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        n_samples = int(self.dataset_size) + int(target.size()[0])
        # L2_distance = L1_distance(total0, total1) # 计算高斯核中的|x-y|
        L2_distance = self.get_distance(target)

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # 将多个核合并在一起

    def mmd(self, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        start_time = time.time()

        n = self.dataset_size
        m = int(target.size()[0])

        kernels = self.guassian_kernel(target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

        loss = (XX + XY).sum() + (YX + YY).sum()
        end_time = time.time()
        # print("mmd time: ", end_time - start_time)
        return loss

    def mmd_for_data_set(self, target_index):
        return self.mmd(target_index).cpu().item()


if __name__ == "__main__":

    # 创建一个张量
    # values = torch.tensor([7, 2, 5, 1, 9, 3])

    #
    # # 对张量进行排序并取出前n个数
    # top_values, top_indices = torch.topk(values, k=3, largest=False)
    #
    # print("排序后的值：", top_values)
    # print("对应的索引：", top_indices)
    # top_value = top_values.max() + top_values.max() / 1000 if top_values.max() != 0 else top_values.max() + 0.001
    # total_scores = (top_value - top_values).sum()
    # # 构建轮盘
    # roulette_wheel = (top_value - top_values) / total_scores
    # # print(roulette_wheel.max())
    # # print(roulette_wheel.min())
    # # 生成随机数，模拟轮盘赌选择
    # random_number = random.random()
    # # 轮盘赌选择
    # cumulative_prob = 0
    # selected_index = None
    # for i, prob in enumerate(roulette_wheel):
    #     cumulative_prob += prob
    #     if random_number < cumulative_prob:
    #         selected_index = top_indices[i]
    #         break
    # # selected_index = random.choices(top_indices)
    # print("selected", selected_index)
    #
    # exit(0)
    # 样本数量可以不同，特征数目必须相同
    data_train = MNIST('./data',
                       download=False,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor()]))
    data_test = MNIST('./data',
                      train=False,
                      download=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
    # data = data_train.data.reshape(60000, -1)
    # labels = data_train.targets.reshape(60000, -1)
    data = data_train.data
    labels = data_train.targets
    # torch.cuda.set_device(0)
    total_size = 8
    data_1 = data.to(device='cuda')
    # data_2 = data[0:10]
    # data_1 = data[0:60000].to(device='cuda')
    # data_2 = data[0:20000].to(device='cuda')
    mmd = MMD(data_1)
    while True:
        print("start")
        time.sleep(6000)
    exit(0)
    size = 3

    combinations = list(itertools.combinations(list(range(total_size)), size))
    print(combinations)
    min_set = None
    min_value = None
    for i in combinations:
        target = torch.tensor(i)
        dis = mmd.mmd_for_data_set(target)
        print(i, ": ", dis)
        if min_value is None or dis < min_value:
            min_value = dis
            min_set = i
    print("min: ", min_set, min_value)
    subset = torch.tensor([0, 1, 2, 4])
    final_set = mmd.get_min_distance_index_in_range(size, subset)
    print(final_set)
    print(mmd.mmd_for_data_set(torch.tensor(final_set)))

    # data_2 = torch.tensor([0])
    # data_3 = torch.tensor([1])
    # data_4 = torch.tensor([2])
    # data_5 = torch.tensor([3])
    # data_6 = torch.tensor([0, 2, 3])
    # data_7 = torch.tensor([0, 1, 3])
    # data_8 = torch.tensor([0, 1, 2])
    # data_9 = torch.tensor([1, 2, 3])
    #
    # data_0_1 = torch.tensor([0, 1])
    # data_0_2 = torch.tensor([0, 2])
    # data_0_3 = torch.tensor([0, 3])
    # data_1_2 = torch.tensor([1, 2])
    # data_1_3 = torch.tensor([1, 3])
    # data_2_3 = torch.tensor([2, 3])
    # print("0_1:", mmd.mmd_for_data_set(data_0_1))
    # print("0_2:", mmd.mmd_for_data_set(data_0_2))
    # print("0_3:", mmd.mmd_for_data_set(data_0_3))
    # print("1_2:", mmd.mmd_for_data_set(data_1_2))
    # print("1_3:", mmd.mmd_for_data_set(data_1_3))
    # print("2_3:", mmd.mmd_for_data_set(data_2_3))
    # data_2 = torch.tensor([i for i in range(100)])
    # data_3 = torch.tensor([i+200 for i in range(100)])
    # print("0:", mmd.mmd_for_data_set(data_2))
    # print("1:", mmd.mmd_for_data_set(data_3))
    # print("2:", mmd.mmd_for_data_set(data_4))
    # print("3:", mmd.mmd_for_data_set(data_5))
    # print("-0:", mmd.mmd_for_data_set(data_9))
    # print("-1:", mmd.mmd_for_data_set(data_6))
    # print("-2:", mmd.mmd_for_data_set(data_7))
    # print("-3:", mmd.mmd_for_data_set(data_8))

    exit(0)
