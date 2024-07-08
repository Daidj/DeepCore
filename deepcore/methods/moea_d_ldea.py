import math
import os
import random
import time
import pickle
from collections import OrderedDict
from functools import cmp_to_key

import copy
from cleanlab.filter import find_label_issues

from cleanlab import Datalab
from scipy.stats import gaussian_kde, entropy
from numpy import linspace
from torch.utils.data import TensorDataset

from mmd_algorithm import MMD
from .earlytrain import EarlyTrain
import torch
import numpy as np
import matplotlib.pyplot as plt

from .methods_utils import *

MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect
test_data_folder = 'test_data'





def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] - arr[j + 1] >= 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def plot_nested_list(nested_list, diff=None, important_points=None, title='X-Y', folder_name='test_data'):
    x = [point[0] for point in nested_list]
    y = [point[1] for point in nested_list]
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, s=5, c='b')
    # for i, j in zip(x, y):
    #     plt.annotate(f'({i:.2f},{j:.2f})', (i, j))
    if diff != None:
        diff_x = [point[0] for point in diff]
        diff_y = [point[1] for point in diff]
        plt.scatter(diff_x, diff_y, s=5, c='r')
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'({i:.4f},{j:.4f})', (i, j), color='red')

    if important_points != None:
        important_x = [point[0] for point in important_points]
        important_y = [point[1] for point in important_points]
        plt.scatter(important_x, important_y, s=5, c='g')
        for i, j in zip(important_x, important_y):
            plt.annotate(f'({i:.4f},{j:.4f})', (i, j), color='green')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, '{}.png'.format(title))
    plt.savefig(file_path)
    plt.show()


class Individual:
    fitness_calculators = None
    device = 'cuda'
    last_init_individual = None

    def __init__(self, total_gene_num, gene_num, target_num):
        self.total_gene_num = total_gene_num
        self.gene_num = gene_num
        self.gene = set()
        self.unselected_gene = set(torch.arange(total_gene_num).numpy())
        self.target_num = target_num
        self.fitness = None

    def clone(self):
        copy_ind = copy.deepcopy(self)
        return copy_ind

    def crossover(self, other):
        if self.gene == other.gene:
            return self.mutation(), other.mutation()
        child_1 = self.clone()
        gene_1 = random.choice(list(self.gene - other.gene))
        child_2 = other.clone()
        gene_2 = random.choice(list(other.gene - self.gene))
        child_1.gene.remove(gene_1)
        child_1.gene.add(gene_2)
        child_1.unselected_gene.remove(gene_2)
        child_1.unselected_gene.add(gene_1)
        child_1.set_fitness()
        child_2.gene.remove(gene_2)
        child_2.gene.add(gene_1)
        child_2.unselected_gene.remove(gene_1)
        child_2.unselected_gene.add(gene_2)
        child_2.set_fitness()
        return child_1, child_2

    def mutation(self):

        child = self.clone()
        new_gene = random.choice(list(self.unselected_gene))
        remove_gene = random.choice(list(self.gene))
        child.gene.remove(remove_gene)
        child.gene.add(new_gene)
        child.unselected_gene.remove(new_gene)
        child.unselected_gene.add(remove_gene)
        child.set_fitness()
        return child

    def random_init(self):
        self.gene = set(random.sample(self.unselected_gene, self.gene_num))
        self.unselected_gene = self.unselected_gene - self.gene
        self.set_fitness()

    def local_search(self, weight_vector):
        print(self.fitness, " local search: ")
        child = self.clone()
        child.__remove_worst(weight_vector)
        child.__greedy_search(weight_vector, 1)
        child.set_fitness()
        if child < self:
            print("search better")
        elif child > self:
            print("search worse")
        else:
            print("search equal")
        return child

    def local_search_old(self, weight_vector):
        print(self.fitness, " local search: ")
        child = self.clone()
        num = math.ceil(self.gene_num * 0.01)
        remove_gene = set(random.sample(list(self.gene), num))
        child.gene = child.gene - remove_gene
        child.unselected_gene.update(remove_gene)
        child.__greedy_search(weight_vector, num)
        child.set_fitness()
        if child < self:
            print("search better")
        elif child > self:
            print("search worse")
        else:
            print("search equal")
        return child

    def __remove_worst(self, weight_vector):
        score_array = torch.stack([c.selected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
            self.device)
        res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
        res = res.squeeze(1)
        selected = list(self.gene)[torch.argmax(res)]
        self.gene.discard(selected)
        self.unselected_gene.add(selected)

    def __greedy_search(self, weight_vector, num):

        if len(self.gene) == 0:
            selected = random.choice(list(self.unselected_gene))
            self.gene.add(selected)
            self.unselected_gene.discard(selected)
            num -= 1
        for i in range(num):
            score_array = torch.stack([c.unselected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
                self.device)
            res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
            res = res.squeeze(1)
            selected = list(self.unselected_gene)[torch.argmin(res)]
            self.gene.add(selected)
            self.unselected_gene.discard(selected)

    def greedy_init(self, weight_vector, fraction=None):
        init_num = self.gene_num
        if fraction is not None:
            sample_num = round(fraction * self.gene_num)
            self.gene = set(random.sample(list(Individual.last_init_individual), sample_num))
            self.unselected_gene = self.unselected_gene - self.gene
            init_num = self.gene_num - sample_num
        self.__greedy_search(weight_vector, init_num)

        Individual.last_init_individual = self.gene
        print('init finish: ', fraction)
        self.set_fitness()

    def init(self, selected):
        self.gene = set(selected)
        self.unselected_gene = self.unselected_gene - self.gene
        self.set_fitness()

    def __eq__(self, other):
        if self > other or self < other:
            return False
        else:
            return True

    def __lt__(self, other):
        for i in range(self.target_num):
            if self.fitness[i] >= other.fitness[i]:
                return False
        return True

    def __gt__(self, other):
        for i in range(self.target_num):
            if self.fitness[i] <= other.fitness[i]:
                return False
        return True

    def get_single_fitness(self, weight_vector):

        dot_product = torch.dot(weight_vector.cpu(), torch.tensor(self.fitness))

        return dot_product.item()

    def set_fitness(self):
        self.fitness = [round(calculator.fitness(self), 4) for calculator in self.fitness_calculators]
    # def compare_individual(individual_1, individual_2):
    #     if all(a <= b for a, b in zip(individual_1, individual_2)):
    #         return -1
    #     else:
    #         return 1


class SubProblems:
    def __init__(self, target_num, count, device):
        self.target_num = target_num
        # self.unit_vectors = torch.eye(self.target_num).to(device)
        self.unit_vectors = torch.cat([torch.eye(self.target_num), torch.tensor([[1 / 2 ** 0.5, 1 / 2 ** 0.5]])],
                                      dim=0).to(device)

        self.regions = []
        self.weight_vectors = self.build_subproblem(target_num, count).to(device)
        self.distance = euclidean_dist(self.weight_vectors, self.weight_vectors)
        self.set_neighbors()
        self.decomposition()

    def build_subproblem(self, target_num, count):
        problems = []
        min_fraction = 1e-4
        max_fraction = 0.9999
        if target_num == 2:
            step = (max_fraction - min_fraction) / (count - 1)
            for i in range(count):
                weight_vector = [min_fraction + i * step, 1 - min_fraction - i * step]
                problems.append(weight_vector)
        else:
            # TODO:随机初始化权重向量
            pass
        return torch.tensor(problems)

    def set_neighbors(self, W=3):
        # 相邻子问题的数量
        _, indices = torch.topk(self.distance, k=W + 1, dim=1, largest=False)
        indices = indices.cpu().numpy()
        self.neighbors = []
        for i in range(indices.shape[0]):
            neighbor = set(indices[i])
            neighbor.discard(i)
            self.neighbors.append(neighbor)

    def decomposition(self):
        # (50, 2) . (2, 2) => (50, 2)
        dot_products = torch.matmul(self.weight_vectors, self.unit_vectors.T)
        max_indices = torch.argmax(dot_products, dim=1)
        for i in range(self.target_num):
            indices = torch.where(max_indices == i)[0]
            indices = indices.cpu().numpy()
            self.regions.append(set(indices))
        # print(self.regions)


class MODE:
    def __init__(self, fitness_calculators: list, total_gene_num: int, budget: int, device, population_num=50,
                 output_folder='test_data'):
        self.population_num = population_num
        self.total_gene_num = total_gene_num
        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > self.total_gene_num:
            budget = self.total_gene_num
        self.gene_num = budget
        self.target_num = len(fitness_calculators)
        self.subproblems = SubProblems(target_num=self.target_num, count=population_num, device=device)
        self.best_population_for_subproblems = []
        Individual.fitness_calculators = fitness_calculators
        for i in range(population_num):
            individual = Individual(total_gene_num=self.total_gene_num, gene_num=self.gene_num,
                                    target_num=self.target_num)
            if i % 10 == 0:
                individual.greedy_init(self.subproblems.weight_vectors[i])
            else:
                individual.greedy_init(self.subproblems.weight_vectors[i], 0.9 - (i % 10) / 20)
            self.best_population_for_subproblems.append(individual)
        self.best_population_for_pareto = self.best_population_for_subproblems

        # self.best_population_for_pareto = bubble_sort(self.best_population_for_pareto)
        # sorted(self.best_population_for_pareto)

        self.best_solution = []
        self.best_solution_to_subproblem = []
        self.last_best_index = -1
        self.count_set = np.ones(self.population_num)
        for i in range(population_num):
            current = self.best_population_for_subproblems[i]
            nondeminated = True
            for other in self.best_population_for_subproblems:
                if current > other:
                    nondeminated = False
                    break
            if nondeminated:
                self.count_set[i] = self.count_set[i] + 1
                self.best_solution.append(current)
                self.best_solution_to_subproblem.append(i)
        self.device = device
        self.greedy_best = []
        for calculator in fitness_calculators:
            i = Individual(self.total_gene_num, self.gene_num, self.target_num)
            i.init(calculator.get_best())
            self.greedy_best.append(i)
        self.greedy_best_fitness_points = [p.fitness for p in self.greedy_best]
        self.output_folder = output_folder

    def get_best_in_solution(self):
        # fitness_front = [p.fitness for p in self.best_solution]
        # front_tensor = torch.tensor(fitness_front)
        # greedy_best = torch.tensor(self.greedy_best_fitness_points)
        # scores = len(greedy_best) * front_tensor - torch.sum(greedy_best, dim=0)
        # scores = torch.sum(scores, dim=1)
        # best = torch.argmin(scores)

        # fitness_front = [p.fitness for p in self.best_solution]
        # front_tensor = torch.tensor(fitness_front)
        # greedy_best = torch.tensor(self.greedy_best_fitness_points)
        # min_v, _ = torch.min(greedy_best, dim=0)
        # scores = front_tensor - min_v
        # scores[:, -1] = scores[:, -1] * 0.5
        # scores = torch.sum(scores, dim=1)
        # best = torch.argmin(scores)

        # fitness_front = [p.fitness for p in self.best_solution]
        # front_tensor = torch.tensor(fitness_front)
        # best = torch.argmin(front_tensor[:, -1])

        fitness_front = [p.fitness for p in self.best_solution]
        front_tensor = torch.tensor(fitness_front)
        greedy_best = torch.tensor(self.greedy_best_fitness_points)
        best_point = torch.min(greedy_best, dim=0).values
        worst_point = torch.max(greedy_best, dim=0).values
        scores = torch.sum((front_tensor - best_point) / (worst_point - best_point), dim=1)
        best = torch.argmin(scores)
        return best

    def solve(self, iter=50):
        # 子空间数量
        subregions_num = self.target_num
        L = 10
        utility = np.ones((self.population_num, L))

        beta = min(self.target_num / 10, 0.8)
        for i in range(iter):
            print("Iter:", i)
            for opr in range(self.population_num):
                utility[:, i % L] = 1
                regions = self.subproblems.regions
                local_probability = np.array([])
                for region in regions:
                    local_probability = np.append(local_probability, 1 / self.count_set[list(region)].sum())
                selected_region = np.argmax(local_probability)
                if random.random() < beta:

                    selected_subproblem = random.choice(list(self.subproblems.regions[selected_region]))

                else:
                    utility_sum = np.sum(utility, axis=1)
                    selected_subproblem = np.argmax(utility_sum)
                print('subprobleam: ', selected_subproblem)
                parent_1 = self.best_population_for_subproblems[selected_subproblem]
                neighboring = random.choice(list(self.subproblems.neighbors[selected_subproblem]))
                parent_2 = self.best_population_for_subproblems[neighboring]
                child_1, child_2 = parent_1.crossover(parent_2)
                child_3 = parent_1.mutation()
                child_4 = parent_2.mutation()
                # child_5 = parent_1.local_search(self.subproblems.weight_vectors[selected_subproblem])
                # child_6 = parent_2.local_search(self.subproblems.weight_vectors[neighboring])
                # new_population = [parent_1, parent_2, child_1, child_2, child_3, child_4, child_5, child_6]
                new_population = [parent_1, parent_2, child_1, child_2, child_3, child_4]
                new_population_subproblems = [selected_subproblem if i % 2 == 0 else neighboring for i in
                                              range(len(new_population))]
                new_population.append(self.best_solution[self.last_best_index].local_search(
                    self.subproblems.weight_vectors[self.best_solution_to_subproblem[self.last_best_index]]))
                new_population_subproblems.append(self.best_solution_to_subproblem[self.last_best_index])
                single_objective_1 = np.array(
                    [p.get_single_fitness(self.subproblems.weight_vectors[selected_subproblem]) for p in
                     new_population])
                self.best_population_for_subproblems[selected_subproblem] = new_population[
                    np.argmin(single_objective_1)]
                single_objective_2 = np.array(
                    [p.get_single_fitness(self.subproblems.weight_vectors[neighboring]) for p in new_population])
                self.best_population_for_subproblems[neighboring] = new_population[np.argmin(single_objective_2)]

                for j in range(2, len(new_population)):
                    current = new_population[j]
                    nondeminated = True
                    replaced = False
                    for k in range(len(self.best_solution)):
                        if current > self.best_solution[k]:
                            nondeminated = False
                            break
                        elif current < self.best_solution[k]:
                            self.best_solution[k] = current
                            self.best_solution_to_subproblem[k] = new_population_subproblems[j]
                            replaced = True
                            break
                    if nondeminated:
                        if not replaced:
                            self.best_solution.append(current)
                            self.best_solution_to_subproblem.append(new_population_subproblems[j])
                        self.count_set[selected_subproblem] = self.count_set[selected_subproblem] + 1
                    if not current > new_population[0]:
                        utility[selected_subproblem][i % L] = utility[selected_subproblem][i % L] + 1

            fitness_front = [p.fitness for p in self.best_solution]
            subproblems_front = [p.fitness for p in self.best_population_for_subproblems]
            best = self.get_best_in_solution()
            self.last_best_index = best
            if i % 10 == 0:
                # plot_nested_list(subproblems_front, diff=self.greedy_best_fitness_points, title="Iter_subproblems_{}".format(i), folder_name=self.output_folder)
                plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, title="Iter_pareto_{}".format(i),
                                 important_points=[self.best_solution[best].fitness], folder_name=self.output_folder)

        final_best_solution = []
        for i in range(len(self.best_solution)):
            current = self.best_solution[i]
            nondeminated = True
            for other in self.best_solution:
                if current > other:
                    nondeminated = False
                    break
            if nondeminated:
                final_best_solution.append(current)
        self.best_solution = final_best_solution
        # 对帕累托前沿的点进行评分并选择最优解
        best = self.get_best_in_solution()
        best_individual = self.best_solution[best]
        best_fitness = best_individual.fitness
        print("best fitness: ", best_fitness)
        plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, important_points=[best_fitness],
                         title="final_pareto", folder_name=self.output_folder)
        print("fitness front: ", len(fitness_front))
        return np.array(list(best_individual.gene)), best_fitness





class MOEA(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Confidence",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

        self.epochs = epochs
        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def write_file(self, class_index, scores, id, name='class'):
        folder_name = 'test_data'
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, '{}_{}.pkl'.format(name, id))
        my_array = class_index[np.argsort(scores[-1])]
        with open(file_path, 'wb') as file:
            pickle.dump(my_array, file)
        # with open(file_path, 'rb') as file:
        #     loaded_array = pickle.load(file)

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        scores = np.array([])
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                                          torch.utils.data.Subset(self.dst_train, index),
                                                          batch_size=self.args.selection_batch,
                                                          num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    outputs = self.model(inputs.to(self.args.device))
                    matrix.append(self.model.embedding_recorder.embedding)
                    if self.selection_method == "LeastConfidence":
                        scores = np.append(scores, outputs.max(axis=1).values.cpu().numpy())
                    elif self.selection_method == "Entropy":
                        preds = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                    elif self.selection_method == "Confidence":
                        preds = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        max_preds = preds.max(axis=1)
                        # print(max_preds)
                        # print(max_preds.shape)
                        scores = np.append(scores, max_preds)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0), scores

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            scores = []
            # if type(self.dst_train.data) == torch.Tensor:
            #     data = self.dst_train.data.reshape(self.dst_train.data.shape[0], -1).to(device='cuda')
            # elif type(self.dst_train.data) == np.ndarray:
            #     data = torch.from_numpy(self.dst_train.data.reshape(self.dst_train.data.shape[0], -1)).to(device='cuda')
            for c in range(self.args.num_classes):
                test_data_folder = 'test_data/{}'.format(c)
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                features_matrix, importance = self.construct_matrix(class_index)
                size = round(len(class_index) * self.fraction)
                time1 = time.time()
                fitness_calculators = [RepresentativenessCalculator(features_matrix, size, device='cuda'),
                                       DiversityCalculator(features_matrix, size, device='cuda')]
                # fitness_calculators = [UniquenessCalculator(importance, size, device='cuda'),
                #                        RepresentativenessCalculator(features_matrix, size, device='cuda')]
                # fitness_calculators = [UniquenessCalculator(importance, size, device='cuda'),
                #                        DiversityCalculator(features_matrix, size, device='cuda')]
                solver = MODE(fitness_calculators=fitness_calculators, total_gene_num=len(class_index), budget=size,
                              device='cuda',
                              population_num=20, output_folder=test_data_folder)
                res_heuristic, heuristic_fitness = solver.solve(iter=50)
                time2 = time.time()
                print("heuristic fitness: ", heuristic_fitness)
                print("heuristic time: ", time2 - time1)
                best_result = class_index[res_heuristic]
                print("best size:", len(best_result))
                selection_result = np.append(selection_result, best_result)
        else:
            pass
            scores = self.rank_uncertainty()
            selection_result = np.argsort(scores)[:self.coreset_size]

        return {"indices": selection_result, "scores": scores}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
