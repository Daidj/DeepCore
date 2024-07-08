import math
import os
import random
import time
import pickle
from collections import OrderedDict
from functools import cmp_to_key

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

MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect

# class Solver_old:
#     def __init__(self, matrix, budget: int, metric, device):
#         if type(matrix) == torch.Tensor:
#             assert matrix.dim() == 2
#         elif type(matrix) == np.ndarray:
#             assert matrix.ndim == 2
#             matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
#         self.features_matrix = matrix.to(device)
#         self.total_num = matrix.shape[0]
#         if budget < 0:
#             raise ValueError("Illegal budget size.")
#         elif budget > self.total_num:
#             budget = self.total_num
#         self.budget = budget
#         assert callable(metric)
#         self.metric = metric
#         self.device = device
#         self.all_samples = torch.arange(self.total_num)
#         self.distance = metric(self.features_matrix, self.features_matrix).to(self.device)
#
#     def fitness(self, solution):
#         not_selected = torch.tensor([x for x in self.all_samples if x not in solution])
#         # print(self.distance)
#         dis = self.distance[not_selected, :][:, solution]
#         # print(dis)
#         min_dis, _ = torch.min(dis, dim=1)
#         # total_distance = np.sum(min_dis)
#         # # 越小说明解越好
#         # return total_distance
#         max_dis = torch.max(min_dis)
#         # 解越小说明解越好
#         return max_dis.item()
#
#     def fitness2(self):
#         return 0
#
#     def new_solution(self, solution, n):
#         not_selected = torch.tensor([x for x in self.all_samples if x not in solution])
#         indices = random.sample(range(len(solution)), self.budget-n)
#         reserve = solution[indices]
#         indices_arr2 = random.sample(range(len(not_selected)), n)
#         new = not_selected[indices_arr2]
#         new_arr = torch.cat([reserve, new])
#         # print(new_arr)
#         return new_arr
#
#     def heuristic_solver(self, random_seed=None, index=None, already_selected=None):
#         if index is None:
#             best_solution = self.all_samples[random.sample(range(self.total_num), self.budget)]
#         else:
#             if type(index) == np.ndarray:
#                 index = torch.from_numpy(index).requires_grad_(False)
#             best_solution = index
#         best_fitness = self.fitness(best_solution)
#         history_best = best_solution
#         history_fitness = best_fitness
#         T_init = self.budget/3
#
#         alpha = 0.95
#         T_min = T_init * math.pow(alpha, 100)
#         L = 50 # 每个温度下的迭代次数
#
#         fitness_all = []
#         fitness_all.append(best_fitness)
#         T = T_init
#         balance = True
#         while T > T_min:
#             print('Iter:', T)
#             for i in range(L):
#                 new_solution = self.new_solution(best_solution, int(T))
#                 new_fitness = self.fitness(new_solution)
#                 if balance:
#                     delta_e = abs(new_fitness - best_fitness)
#                     T_init = max(2*delta_e, 10)
#                     T_min = max(T_init * math.pow(alpha, 100), 0.9)
#                     T = T_init
#                     balance = False
#                     continue
#                 if new_fitness <= best_fitness:
#                     best_solution = new_solution
#                     best_fitness = new_fitness
#                 else:
#                     delta_e = abs(new_fitness-best_fitness)
#                     p_accept = math.exp(-delta_e / T)
#                     if random.random() < p_accept:
#                         best_solution = new_solution
#                         best_fitness = new_fitness
#                 if best_fitness < history_fitness:
#                     history_best = best_solution
#                     history_fitness = best_fitness
#                 fitness_all.append(best_fitness)
#             T = alpha * T
#         fig, ax = plt.subplots()
#         ax.plot(fitness_all)
#         ax.set_title('List as a Plot')
#         ax.set_xlabel('Index')
#         ax.set_ylabel('Value')
#         plt.show()
#         return history_best

def compare_individual(individual_1, individual_2):
    return individual_1[0]-individual_2[0]

# def compare_individual(individual_1, individual_2):
#     if all(a <= b for a, b in zip(individual_1, individual_2)):
#         return -1
#     else:
#         return 1

class Solver:
    def __init__(self, matrix, importance, budget: int, metric, device, population_num=50):
        if type(matrix) == torch.Tensor:
            assert matrix.dim() == 2
        elif type(matrix) == np.ndarray:
            assert matrix.ndim == 2
            matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
        if type(importance) == np.ndarray:
            importance = torch.from_numpy(importance).requires_grad_(False).to(device)
        self.features_matrix = matrix.to(device)
        self.importance = importance.to(device)
        self.total_num = matrix.shape[0]
        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > self.total_num:
            budget = self.total_num
        self.gene_num = budget
        assert callable(metric)
        self.metric = metric
        self.device = device
        self.all_samples = torch.arange(self.total_num)
        self.distance = metric(self.features_matrix, self.features_matrix).to(self.device)
        self.population = []
        self.population_fitness = []
        self.population_num = population_num

    def fitness(self, solution):
        fitness = [self.fitness1(solution), self.fitness2(solution)]
        return fitness

    def fitness1(self, solution):
        not_selected = torch.tensor([x for x in self.all_samples if x not in solution])
        # print(self.distance)
        dis = self.distance[not_selected, :][:, solution]
        # print(dis)
        min_dis, _ = torch.min(dis, dim=1)
        # total_distance = np.sum(min_dis)
        # # 越小说明解越好
        # return total_distance
        max_dis = torch.max(min_dis)
        # 解越小说明解越好
        return max_dis.item()

    def fitness2(self, solution):
        fitness = torch.sum(self.importance[solution])
        return fitness.item()

    def init_individual(self):
        return self.all_samples[random.sample(range(self.total_num), self.gene_num)]

    def init_population(self):
        start = time.time()
        res_greedy = torch.from_numpy(k_center_greedy(matrix=self.features_matrix, budget=self.gene_num, metric=euclidean_dist, device=self.device))
        greedy_fitness = self.fitness(res_greedy)
        self.population.append(res_greedy)
        self.population_fitness.append(greedy_fitness)
        print("greedy fitness: ", greedy_fitness)
        print("time: ", time.time() - start)
        _, sorted_indices = torch.sort(self.importance)
        res_importance = sorted_indices[:self.gene_num].cpu()
        imp_fitness = self.fitness(res_importance)
        self.population.append(res_importance)
        self.population_fitness.append(imp_fitness)
        print("imp fitness: ", imp_fitness)
        for i in range(self.population_num):
            ind = self.init_individual()
            self.population.append(ind)
            self.population_fitness.append(self.fitness(ind))

    def mate(self, individual_1, individual_2):
        gene = random.randint(0, self.gene_num-1)
        child_1 = individual_1.clone().detach()
        child_2 = individual_2.clone().detach()
        if torch.any(child_1 == individual_2[gene]):
            child_1 = self.mutation(child_1)
        else:
            child_1[gene] = individual_2[gene]
        if torch.any(child_2 == individual_1[gene]):
            child_2 = self.mutation(child_2)
        else:
            child_2[gene] = individual_1[gene]
        return child_1, child_2

    def mutation(self, individual, n=1):
        gene = random.sample(range(self.gene_num), n)
        child = individual.clone().detach()
        not_selected = torch.tensor([x for x in self.all_samples if x not in individual])
        index = random.sample(range(len(not_selected)), n)
        child[gene] = not_selected[index]
        return child

    def new_solution(self, solution, n):
        not_selected = torch.tensor([x for x in self.all_samples if x not in solution])
        indices = random.sample(range(len(solution)), self.gene_num - n)
        reserve = solution[indices]
        indices_arr2 = random.sample(range(len(not_selected)), n)
        new = not_selected[indices_arr2]
        new_arr = torch.cat([reserve, new])
        # print(new_arr)
        return new_arr

    def heuristic_solver(self, iter=100, index=None, already_selected=None):
        self.init_population()
        if index is None:
            best_individual = self.all_samples[random.sample(range(self.total_num), self.gene_num)]
        else:
            if type(index) == np.ndarray:
                index = torch.from_numpy(index).requires_grad_(False)
            best_individual = index
        best_fitness = self.fitness(best_individual)
        fitness_all = []
        fitness_all.append(best_fitness)

        def plot_nested_list(nested_list, title='X-Y'):
            x = [point[0] for point in nested_list]
            y = [point[1] for point in nested_list]
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y)
            plt.xlim(min(x) - 100, max(x) + 100)
            plt.ylim(min(y) - 100, max(y) + 100)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(title)
            folder_name = 'test_data'
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, '{}.png'.format(title))
            plt.savefig(file_path)
            plt.show()

        for i in range(iter):
            print('Iter:', i)
            new_population = []
            new_population.extend(self.population)
            new_population_fitness = []
            for j in range(self.population_num):
                # new_population.append(self.population[j])
                other = random.randint(0, self.population_num-1)
                while j == other:
                    other = random.randint(0, self.population_num-1)
                ind1, ind2 = self.mate(self.population[j], self.population[other])
                new_population.append(ind1)
                new_population.append(ind2)
                mutation_num = max(1, min(int((iter-i)*2), int(self.gene_num*0.05/(i+1))))
                mutation_ind = self.mutation(self.population[j], mutation_num)
                new_population.append(mutation_ind)
            # random.shuffle(new_population)
            for individual in new_population:
                new_population_fitness.append(self.fitness(individual))

            self.population = new_population[:self.population_num]
            self.population_fitness = new_population_fitness[:self.population_num]
            j = self.population_num
            while j < len(new_population):
                for k in range(self.population_num):
                    if compare_individual(new_population_fitness[j], self.population_fitness[k]) <= 0:
                        self.population[k] = new_population[j]
                        self.population_fitness[k] = new_population_fitness[j]
                        print('update')
                        break
                j += 1

            # sorted_indices = sorted(range(len(new_population_fitness)), key=cmp_to_key(lambda p, q: compare_individual(new_population_fitness[p], new_population_fitness[q])))[0:self.population_num]
            # self.population = [new_population[p] for p in sorted_indices]
            # self.population_fitness = [new_population_fitness[p] for p in sorted_indices]
            if compare_individual(self.population_fitness[0], best_fitness) <= 0:
                best_individual = self.population[0]
                best_fitness = self.population_fitness[0]

            fitness_all.append(best_fitness)
        #     if i % 10 == 0:
        #         plot_nested_list(self.population, 'Iter_{}'.format(i))
        # plot_nested_list(fitness_all, 'history_best')
        sorted_indices_1 = sorted(range(len(self.population_fitness)), key=lambda p:new_population_fitness[p][0])
        sorted_indices_2 = sorted(range(len(self.population_fitness)), key=lambda p:new_population_fitness[p][1])
        intersection = OrderedDict.fromkeys(sorted_indices_1[0:int(self.population_num*0.5)]).keys() & OrderedDict.fromkeys(sorted_indices_2[0:int(self.population_num*0.5)]).keys()
        if len(intersection) == 0:
            best_index = 2
        else:
            best_index = list(intersection)[0]
        best_index = 0
        best_individual = self.population[best_index]
        best_fitness = self.population_fitness[best_index]
        return best_individual, best_fitness

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

class Heuristic(EarlyTrain):
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
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                features_matrix, importance = self.construct_matrix(class_index)
                size = round(len(class_index) * self.fraction)
                # 直接并集
                # res = k_center_greedy(matrix=features_matrix, budget=size, metric=euclidean_dist, device="cuda")
                # res2 = np.argsort(importance)[:size]
                # intersection = set(res).intersection(set(res2))
                # print(len(intersection))
                # index = 0
                # while len(intersection) < size:
                #     intersection.add(res[index])
                #     intersection.add(res[index])
                #     index += 1
                time1 = time.time()
                solver = Solver(features_matrix, importance, size, metric=euclidean_dist, device='cuda', population_num=50)
                res_heuristic, heuristic_fitness = solver.heuristic_solver(iter=100)
                time2 = time.time()
                print("heuristic fitness: ", heuristic_fitness)
                print("heuristic time: ", time2-time1)
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
