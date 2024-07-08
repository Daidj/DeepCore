import torch
import numpy as np


def euclidean_dist(x, y):
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


def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected is None:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            already_selected = np.array(already_selected)
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(
            device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])
        # print(dis_matrix)
        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class RepresentativenessCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        distance = metric(self.features_matrix, self.features_matrix)
        mean_distance = torch.mean(distance, dim=1).cpu()
        self.distance = distance.cpu()
        self.min_fitness = torch.min(mean_distance) + 1e-6
        self.max_fitness = torch.max(mean_distance) + 1e-6
        self.samples_fitness = (mean_distance-self.min_fitness)/(self.max_fitness-self.min_fitness)
        self.device = device

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        fitness = torch.mean(self.samples_fitness[selected])
        return fitness.item()

    def unselected_fitness(self, individual):
        unselected = torch.tensor(list(individual.unselected_gene))
        scores = self.samples_fitness[unselected]
        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        fitness = self.samples_fitness[selected]
        return fitness

    def single_local_search(self, individual):
        pass

    def get_best(self):
        _, sorted_indices = torch.sort(self.samples_fitness)
        res_importance = sorted_indices[:self.gene_num]
        return set(res_importance)


class UniquenessCalculator:
    def __init__(self, condidence, gene_num, device, metric=euclidean_dist):
        if type(condidence) == np.ndarray:
            condidence = torch.from_numpy(condidence).requires_grad_(False)
        self.gene_num = gene_num
        self.confidence = condidence
        self.min_fitness = torch.min(self.confidence) + 1e-6
        self.max_fitness = torch.max(self.confidence) + 1e-6
        self.confidence = (self.confidence - self.min_fitness) / (self.max_fitness - self.min_fitness)

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        fitness = torch.mean(self.confidence[selected])
        # normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return fitness.item()

    def unselected_fitness(self, individual):
        unselected = torch.tensor(list(individual.unselected_gene))
        scores = self.confidence[unselected]
        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        fitness = self.confidence[selected]
        # normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return fitness

    def get_best(self):
        _, sorted_indices = torch.sort(self.confidence)
        res_importance = sorted_indices[:self.gene_num]
        return set(res_importance)


class DensityCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        self.distance = metric(self.features_matrix, self.features_matrix).cpu()

        mean_dis = torch.mean(self.distance, dim=1)
        dis, center = mean_dis.min(dim=0)
        self.center = center.item()
        self.mean_distance = dis
        self.min_fitness = 1e-6
        max_fitness = torch.max(self.distance[self.center])
        self.max_fitness = max_fitness.item() + 1e-6
        self.device = device

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[self.center][selected]
        mean_dis = torch.mean(dis)
        fitness = torch.abs(self.mean_distance-mean_dis)
        # 值越小说明解越好
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness.item()

    def unselected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        min_dis, _ = torch.min(dis, dim=1)
        scores = 1 - (min_dis - self.min_fitness) / (self.max_fitness - self.min_fitness)

        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[:][:, selected]
        indices = torch.argmin(dis, dim=1)
        unique_elements, counts = torch.unique(indices, return_counts=True)
        assert len(unique_elements) == individual.gene_num
        # min_count_index = torch.argmin(counts)
        # min_count_element = unique_elements[min_count_index]
        scores = torch.zeros(individual.gene_num)
        for i in range(len(unique_elements)):
            scores[unique_elements[i]] = 1 / counts[i]
        return scores

    def single_local_search(self, individual):
        pass

    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)

class DiversityCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        self.min_fitness = 1e-6
        max_dis, _ = torch.max(self.distance, dim=1)
        max_fitness = torch.max(max_dis)
        self.max_fitness = max_fitness.item() + 1e-6
        self.device = device

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        # print(dis)
        min_dis, _ = torch.min(dis, dim=1)
        # total_distance = np.sum(min_dis)
        # # 越小说明解越好
        fitness = torch.max(min_dis)
        # 值越小说明解越好
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness.item()

    def fitness_old(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        # print(dis)
        min_dis, _ = torch.min(dis, dim=1)
        # total_distance = np.sum(min_dis)
        # # 越小说明解越好
        fitness = torch.mean(min_dis)
        # 值越小说明解越好
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness.item()

    def unselected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        min_dis, _ = torch.min(dis, dim=1)
        scores = 1 - (min_dis - self.min_fitness) / (self.max_fitness - self.min_fitness)

        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        dis = self.distance[selected, :][:, selected]
        second_min_values = torch.kthvalue(dis, 2, dim=1).values

        scores = 1 - (second_min_values - self.min_fitness)/(self.max_fitness - self.min_fitness)
        return scores

    def single_local_search(self, individual):
        pass

    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)


class UniversalityCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.total_gene_num = matrix.shape[0]
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        self.min_fitness = 1e-6
        self.max_fitness = self.total_gene_num

        self.min_distance = 1e-6
        max_dis, _ = torch.max(self.distance, dim=1)
        max_distance = torch.max(max_dis)
        self.max_distance = max_distance.item() + 1e-6

        self.device = device

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[:][:, selected]
        center_indices = torch.argmin(dis, dim=1)
        unique_elements, counts = torch.unique(center_indices, return_counts=True)
        assert len(unique_elements) == individual.gene_num
        # min_count_index = torch.argmin(counts)
        # min_count_element = unique_elements[min_count_index]
        # scores = torch.zeros(individual.gene_num)
        # for i in range(len(unique_elements)):
        #     scores[unique_elements[i]] = 1 / counts[i]
        #
        #
        # min_dis, _ = torch.min(dis, dim=1)
        fitness = torch.max(counts) - torch.min(counts)
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness.item()

    def unselected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        min_dis, _ = torch.min(dis, dim=1)
        scores = 1 - (min_dis - self.min_distance) / (self.max_distance - self.min_distance)

        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[:][:, selected]
        indices = torch.argmin(dis, dim=1)
        unique_elements, counts = torch.unique(indices, return_counts=True)
        assert len(unique_elements) == individual.gene_num
        # min_count_index = torch.argmin(counts)
        # min_count_element = unique_elements[min_count_index]
        # print(counts)
        scores = torch.zeros(individual.gene_num)
        for i in range(len(unique_elements)):
            scores[unique_elements[i]] = 1 - (counts[i] - 1) / (self.total_gene_num - 1)
        return scores

    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)
