import math
import random

import torch
import numpy as np
from numpy.core.function_base import linspace
from scipy.stats import gaussian_kde, entropy
from deepcore.methods.methods_utils.micro_search import first_stage_search
from deepcore.methods.methods_utils import euclidean_dist_for_batch, euclidean_dist
from mmd_algorithm import MMD


def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
        matrix = matrix.to(device)
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
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
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
        self.samples_fitness = (mean_distance - self.min_fitness) / (self.max_fitness - self.min_fitness)
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
    def __init__(self, condidence, gene_num, device, metric=euclidean_dist_for_batch):
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
        # normalization_fitness = (fitness - self.min_fitness)/(self.max_fitness - self.min_fitness)
        return fitness.item()

    def unselected_fitness(self, individual):
        unselected = torch.tensor(list(individual.unselected_gene))
        scores = self.confidence[unselected]
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        scores = self.confidence[selected]
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        # normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return scores

    def get_best(self):
        _, sorted_indices = torch.sort(self.confidence)
        res_importance = sorted_indices[:self.gene_num]
        return set(res_importance)


class DensityCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        with torch.no_grad():
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
        fitness = torch.abs(self.mean_distance - mean_dis)
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
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        # self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        # torch.cuda.empty_cache()

        self.distance = euclidean_dist_for_batch(self.features_matrix, self.features_matrix).cpu()
        second_min_values = torch.kthvalue(self.distance, 2, dim=1).values

        self.min_fitness = second_min_values.min().item()+1e-6
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
        fitness = torch.max(min_dis) + torch.mean(min_dis)
        # 值越小说明解越好
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness.item()

    def unselected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        dis = self.distance[unselected, :][:, selected]
        min_dis, _ = torch.min(dis, dim=1)
        # scores = 1 - (min_dis - min_dis.min()) / (min_dis.max() - min_dis.min())
        scores = 1 - (min_dis - self.min_fitness) / (self.max_fitness - self.min_fitness)
        # scores = scores*0.8+0.1

        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        dis = self.distance[selected, :][:, selected]
        second_min_values = torch.kthvalue(dis, 2, dim=1).values
        # scores = 1 - (second_min_values - second_min_values.min()) / (second_min_values.max() - second_min_values.min())

        scores = 1 - (second_min_values - self.min_fitness) / (self.max_fitness - self.min_fitness)
        # scores = scores*0.8+0.1

        return scores

    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)


class DiversityCalculator2:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        second_min_values = torch.kthvalue(self.distance, 2, dim=1).values

        self.min_fitness = second_min_values.min().item()+1e-6
        max_dis, _ = torch.max(self.distance, dim=1)
        max_fitness = torch.max(max_dis)
        self.max_fitness = max_fitness.item() + 1e-6
        self.device = device

    def fitness_old(self, individual):
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

    def fitness(self, individual):
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
        scores = 1 - (min_dis - min_dis.min()) / (min_dis.max() - min_dis.min())
        # scores = 1 - (min_dis - self.min_fitness) / (self.max_fitness - self.min_fitness)
        # scores = scores*0.8+0.1

        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        dis = self.distance[selected, :][:, selected]
        second_min_values = torch.kthvalue(dis, 2, dim=1).values
        scores = 1 - (second_min_values - second_min_values.min()) / (second_min_values.max() - second_min_values.min())

        # scores = 1 - (second_min_values - self.min_fitness) / (self.max_fitness - self.min_fitness)
        # scores = scores*0.8+0.1

        return scores


    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)

class UniversalityCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
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


class MMDCalculator:
    def __init__(self, matrix, gene_num, device, metric=euclidean_dist_for_batch):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        self.features_matrix = matrix.to(device)
        self.gene_num = gene_num
        self.calculator = MMD(matrix, device)
        # self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        self.min_fitness = 1e-6
        # max_dis, _ = torch.max(self.calculator.distance, dim=1)
        # max_index = torch.argmax(max_dis).cpu().item()
        max_fitness = self.calculator.mmd_for_data_set(torch.tensor([0]))
        self.max_fitness = 1.0
        self.device = device

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        fitness = self.calculator.mmd_for_data_set(selected)
        normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return normalization_fitness, fitness

    def unselected_fitness(self, individual):
        scores = self.calculator.get_unselected_scores(individual.gene, individual.unselected_gene)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def selected_fitness(self, individual):
        scores = self.calculator.get_selected_scores(individual.gene, individual.unselected_gene)
        # selected = torch.tensor(list(individual.gene))
        # dis = self.calculator.distance[selected, :][:, selected]
        # second_min_values = torch.kthvalue(dis, 2, dim=1).values.cpu()
        #
        # scores = 1 - (second_min_values - second_min_values.min()) / (second_min_values.max() - second_min_values.min())
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def set_min_fitness(self, fitness):
        self.min_fitness = fitness

    def set_max_fitness(self, fitness):
        self.max_fitness = max(1e-6, fitness)

    def get_best(self):
        return set(self.calculator.get_min_distance_index(self.gene_num, step_rate=0.12))


class KLCalculator:
    def __init__(self, confidence, gene_num, device, metric=euclidean_dist_for_batch):
        if type(confidence) == np.ndarray:
            confidence = torch.from_numpy(confidence).requires_grad_(False)
        self.gene_num = gene_num
        self.total_gene_num = len(confidence)
        # print('total gene num: ', self.total_gene_num)
        self.confidence = confidence
        self.origin_pdf = gaussian_kde(confidence)
        self.min_confidence = min(confidence)
        self.max_confidence = max(confidence)
        self.lin = linspace(self.min_confidence, self.max_confidence, 200)
        self.origin_distribution = self.origin_pdf.pdf(self.lin)
        self.probability = torch.tensor(self.origin_pdf.pdf(self.confidence))

        # self.min_fitness = torch.min(self.confidence) + 1e-6
        # self.max_fitness = torch.max(self.confidence) + 1e-6
        # self.confidence = (self.confidence - self.min_fitness) / (self.max_fitness - self.min_fitness)

    def crossover_nearby(self, gene: int, unselected_genes: set):
        prob = self.probability[gene]
        unselected_genes_prob = self.probability[list(unselected_genes)]
        scores = torch.abs(unselected_genes_prob-prob)
        return list(unselected_genes)[torch.argmin(scores)]

    def distribution_divergence(self, p, q):
        # M = (p + q) / 2
        # js = 0.5 * entropy(p, M) + 0.5 * entropy(q, M)
        kl = entropy(p, q)
        return kl

    def fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        pdf = gaussian_kde(self.confidence[selected])
        distribution = pdf.pdf(self.lin)
        fitness = self.distribution_divergence(self.origin_distribution, distribution)
        # normalization_fitness = (fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)
        return min(fitness.item(), 0.9999)

    def unselected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        if len(selected) == 1:
            return torch.rand(len(unselected))
        pdf = gaussian_kde(self.confidence[selected])
        origin_probability = self.probability[unselected]
        current_probability = torch.tensor(pdf.pdf(self.confidence[unselected]))
        scores = current_probability - origin_probability
        scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
        # scores = scores*0.8+0.1
        # scores = (scores*0.75+0.05).mul(self.confidence[unselected])
        return scores

    def selected_fitness(self, individual):
        selected = torch.tensor(list(individual.gene))
        unselected = torch.tensor(list(individual.unselected_gene))
        pdf = gaussian_kde(self.confidence[selected])
        origin_probability = self.probability[selected]
        current_probability = torch.tensor(pdf.pdf(self.confidence[selected]))
        scores = current_probability - origin_probability
        scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))
        # scores = scores*0.8+0.1
        # scores = (scores*0.75+0.05).mul(self.confidence[selected])
        return scores

    def get_best(self):
        selected = set(random.sample(range(self.total_gene_num), self.gene_num))
        return selected

    def get_best_old(self):
        total_size = self.total_gene_num
        iter = 20
        step = round(total_size / iter)
        step_selected = math.floor(self.gene_num/iter)
        selected = set()
        start = 0
        _, sorted_indices = torch.sort(self.confidence)
        sorted_index = sorted_indices.numpy()
        for i in range(iter):
            end = start + step
            if i == iter - 1:
                step_selected = self.gene_num - len(selected)
                end = total_size
            selected_index = np.random.choice(sorted_index[start:end], size=step_selected, replace=False)
            selected.update(set(selected_index))
            start = end
        return selected

class InfoCalculator:
    def __init__(self, matrix, confidence, gene_num, device, metric=euclidean_dist_for_batch):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        assert callable(metric)
        if type(confidence) == np.ndarray:
            confidence = torch.from_numpy(confidence).requires_grad_(False)
        self.gene_num = gene_num
        self.confidence = confidence

        self.confidence = (self.confidence - torch.min(self.confidence)) / (torch.max(self.confidence) - torch.min(self.confidence))

        self.features_matrix = matrix.to(device)
        self.distance = metric(self.features_matrix, self.features_matrix).cpu()
        # torch.cuda.empty_cache()

        self.similarity_redundancy_ratio = 1.0
        # second_min_values = torch.kthvalue(self.distance, 2, dim=1).values

        self.min_normalization_fitness = 0.0 - self.similarity_redundancy_ratio * 1.0 * gene_num
        # max_dis, _ = torch.max(self.distance, dim=1)
        # max_fitness = torch.max(max_dis)
        self.max_normalization_fitness = gene_num
        self.min_fitness = 0.0
        self.max_fitness = 1.0

        _, min_num = first_stage_search(matrix=matrix, budget=gene_num, search=False)
        self.min_num = min_num
        self.device = device

    def fitness(self, individual):
        if self.gene_num <= self.min_num:
            origin_fitness = 0.0
        else:
            selected_tensor = torch.tensor(list(individual.gene))
            unselected_tensor = torch.tensor(list(individual.unselected_gene))
            selected_dis = self.distance[selected_tensor, :][:, selected_tensor]
            selected_min_dis = torch.kthvalue(selected_dis, 2, dim=1).values
            selected_bak_dis = torch.kthvalue(selected_dis, 3, dim=1).values

            result_tensor = torch.where(selected_dis != selected_min_dis, torch.tensor(0.0),
                                        selected_bak_dis - selected_dis)
            other_dis = torch.sum(result_tensor, dim=1)
            redundancy_info = (selected_min_dis - other_dis)
            uncertainty = self.confidence[selected_tensor]

            scores = uncertainty - self.similarity_redundancy_ratio * redundancy_info
            fitness = torch.sum(scores).item()
            # 值越小说明解越好
            origin_fitness = (fitness - self.min_normalization_fitness) / (self.max_normalization_fitness - self.min_normalization_fitness)
        normalization_fitness = (origin_fitness - self.min_fitness) / (self.max_fitness - self.min_fitness)

        return normalization_fitness, origin_fitness

    def unselected_fitness(self, individual):
        if self.gene_num <= self.min_num:
            return torch.zeros(len(individual.unselected_gene))
        selected_tensor = torch.tensor(list(individual.gene))
        unselected_tensor = torch.tensor(list(individual.unselected_gene))

        unselected_dis = self.distance[unselected_tensor, :][:, selected_tensor]
        selected_dis = self.distance[selected_tensor, :][:, selected_tensor]
        selected_min_dis = torch.kthvalue(selected_dis, 2, dim=1).values
        result_tensor = -torch.where(unselected_dis > selected_min_dis, torch.tensor(0.0),
                                     unselected_dis - selected_min_dis)
        self_min_dis, _ = torch.min(unselected_dis, dim=1)

        other_dis = torch.sum(result_tensor, dim=1)
        redundancy_info = self_min_dis - other_dis
        uncertainty = self.confidence[unselected_tensor]

        score = uncertainty - self.similarity_redundancy_ratio * redundancy_info
        score = (score - score.min()) / (score.max() - score.min())

        return score

    def selected_fitness(self, individual):
        if self.gene_num <= self.min_num:
            return torch.zeros(len(individual.gene))
        selected_tensor = torch.tensor(list(individual.gene))
        unselected_tensor = torch.tensor(list(individual.unselected_gene))
        unselected_dis = self.distance[unselected_tensor, :][:, selected_tensor]
        selected_dis = self.distance[selected_tensor, :][:, selected_tensor]
        selected_min_dis = torch.kthvalue(selected_dis, 2, dim=1).values
        selected_bak_dis = torch.kthvalue(selected_dis, 3, dim=1).values

        result_tensor = torch.where(selected_dis != selected_min_dis, torch.tensor(0.0),
                                    selected_bak_dis - selected_dis)
        other_dis = torch.sum(result_tensor, dim=1)
        redundancy_info = (selected_min_dis - other_dis)
        uncertainty = self.confidence[selected_tensor]

        score = uncertainty - self.similarity_redundancy_ratio * redundancy_info
        score = (score - score.min()) / (score.max() - score.min())

        # scores = 1 - (second_min_values - self.min_fitness) / (self.max_fitness - self.min_fitness)
        # scores = scores*0.8+0.1

        return score

    def set_min_fitness(self, fitness):
        self.min_fitness = fitness

    def set_max_fitness(self, fitness):
        self.max_fitness = max(1e-6, fitness)

    def get_best(self):
        res_greedy = torch.from_numpy(
            k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist,
                            device=self.device))
        return set(res_greedy)