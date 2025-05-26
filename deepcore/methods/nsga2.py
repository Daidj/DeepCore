import copy
import os
import pickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from .earlytrain import EarlyTrain
from .methods_utils import *
from ..nets.nets_utils import MyDataParallel

MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect
test_data_folder = 'test_data'


def plot_nested_list(nested_list, diff=None, important_points=None, title='X-Y', folder_name='test_data'):
    if len(nested_list[0]) != 2:
        return
    x = [point[0] for point in nested_list]
    y = [point[1] for point in nested_list]
    plt.figure(figsize=(8, 6))
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    plt.scatter(x, y, s=5, c='b')
    # for i, j in zip(x, y):
    #     plt.annotate(f'({i:.2f},{j:.2f})', (i, j))
    if diff != None:
        diff_x = [point[0] for point in diff]
        diff_y = [point[1] for point in diff]
        min_x = min(min_x, min(diff_x))
        max_x = max(max_x, max(diff_x))
        min_y = min(min_y, min(diff_y))
        max_y = max(max_y, max(diff_y))
        plt.scatter(diff_x, diff_y, s=5, c='r')
        for i, j in zip(diff_x, diff_y):
            plt.annotate(f'({i:.5f},{j:.5f})', (i, j), color='red')

    if important_points != None:
        important_x = [point[0] for point in important_points]
        important_y = [point[1] for point in important_points]
        min_x = min(min_x, min(important_x))
        max_x = max(max_x, max(important_x))
        min_y = min(min_y, min(important_y))
        max_y = max(max_y, max(important_y))
        plt.scatter(important_x, important_y, s=5, c='g')
        for i, j in zip(important_x, important_y):
            plt.annotate(f'({i:.5f},{j:.5f})', (i, j), color='green')

    plt.xlim(min_x - 0.2 * (max_x - min_x), max_x + 0.2 * (max_x - min_x))
    plt.ylim(min_y - 0.2 * (max_y - min_y), max_y + 0.2 * (max_y - min_y))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, '{}.png'.format(title))
    plt.savefig(file_path)
    plt.show()
    # plt.close()


class NSGAIndividual:
    fitness_calculators = None
    device = 'cuda'

    def __init__(self, total_gene_num, gene_num, target_num, step_rate=0.01):
        self.total_gene_num = total_gene_num
        self.gene_num = gene_num
        self.gene = set()
        self.unselected_gene = set(torch.arange(total_gene_num).numpy())
        self.target_num = target_num
        self.step_rate = step_rate
        self.fitness = [None for i in range(len(NSGAIndividual.fitness_calculators))]
        self.origin_fitness = [None for i in range(len(NSGAIndividual.fitness_calculators))]

        self.dominate_set = []
        self.dominated_num = 0
        self.rank = 0
        self.distance = 0

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
        print(self.fitness, " local search: ", weight_vector)
        self.step_rate = self.step_rate * 0.9
        print('step_rate: ', self.step_rate)
        search_num = max(1, round(self.step_rate * self.gene_num))
        child = self.clone()
        child.__remove_worst(weight_vector, search_num)
        child.__greedy_search(weight_vector, search_num)
        child.set_fitness()
        if child < self:
            print("search better")
        elif child > self:
            print("search worse")
        else:
            print("search equal")
        return child

    def local_search_random(self, weight_vector):
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

    def __remove_worst(self, weight_vector, num=1):
        score_array = torch.stack([c.selected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
            self.device)
        res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
        res = res.squeeze(1)
        _, indices = torch.topk(res, k=num)
        l = list(self.gene)
        selected = set([l[i] for i in indices])
        self.gene.difference_update(selected)
        self.unselected_gene.update(selected)

    def __remove_worst_single(self, weight_vector):
        score_array = torch.stack([c.selected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
            self.device)
        res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
        res = res.squeeze(1)
        selected = list(self.gene)[torch.argmax(res)]
        self.gene.discard(selected)
        self.unselected_gene.add(selected)

    def __greedy_search(self, weight_vector, num):

        if len(self.gene) <= 5:
            selected = set(random.sample(list(self.unselected_gene), num))
            self.gene.update(selected)
            self.unselected_gene.difference_update(selected)
            return
        score_array = torch.stack([c.unselected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
            self.device)
        res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
        res = res.squeeze(1)
        _, indices = torch.topk(res, k=num, largest=False)
        l = list(self.unselected_gene)
        selected = set([l[i] for i in indices])
        self.gene.update(selected)
        self.unselected_gene.difference_update(selected)

    def __greedy_search_old(self, weight_vector, num):

        if len(self.gene) == 0:
            selected = random.choice(list(self.unselected_gene))
            self.gene.add(selected)
            self.unselected_gene.remove(selected)
            num -= 1
        for i in range(num):
            score_array = torch.stack([c.unselected_fitness(self).float() for c in self.fitness_calculators], dim=0).to(
                self.device)
            res = torch.matmul(score_array.T, weight_vector.unsqueeze(1))
            res = res.squeeze(1)
            selected = list(self.unselected_gene)[torch.argmin(res)]
            self.gene.add(selected)
            self.unselected_gene.remove(selected)

    def greedy_init(self, weight_vector, fraction=None):
        init_num = self.gene_num
        if fraction is not None:
            sample_num = round(fraction * self.gene_num)
            self.gene = set(random.sample(list(NSGAIndividual.last_init_individual), sample_num))
            self.unselected_gene = self.unselected_gene - self.gene
            init_num = self.gene_num - sample_num
        current_rate = 1.0
        while init_num > 0:
            step = min(init_num, round(self.total_gene_num * 0.001))
            step = max(1, step)
            self.__greedy_search(weight_vector, step)
            init_num -= step
            current_rate = current_rate * 0.9

        NSGAIndividual.last_init_individual = self.gene
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
        for i in range(len(self.fitness_calculators)):
            f, origin_f = self.fitness_calculators[i].fitness(self)
            self.fitness[i] = round(f, 6)
            self.origin_fitness[i] = round(origin_f, 6)
    # def compare_individual(individual_1, individual_2):
    #     if all(a <= b for a, b in zip(individual_1, individual_2)):
    #         return -1
    #     else:
    #         return 1


class NSGA2Alg:
    def __init__(self, fitness_calculators: list, total_gene_num: int, budget: int, device, population_num=20,
                 output_folder='test_data', solution_num=5, step_rate=0.01):
        self.population_num = population_num
        self.total_gene_num = total_gene_num
        self.solution_num = solution_num
        self.step_rate = step_rate
        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > self.total_gene_num:
            budget = self.total_gene_num
        self.gene_num = budget
        self.fraction = round(budget / total_gene_num, 2)
        self.target_num = len(fitness_calculators)


        self.best_solution = []

        NSGAIndividual.fitness_calculators = fitness_calculators
        for i in range(population_num):
            individual = NSGAIndividual(total_gene_num=self.total_gene_num, gene_num=self.gene_num,
                                    target_num=self.target_num)
            individual.random_init()
            self.best_solution.append(individual)

        self.fast_non_dominated_sort(self.best_solution)

        self.device = device
        self.greedy_best = []

        self.greedy_best_fitness_points = None
        self.output_folder = output_folder

    def fast_non_dominated_sort(self, P):
        """
        非支配排序
        :param P: 种群 P
        :return F: F=(F_1, F_2, ...) 将种群 P 分为了不同的层， 返回值类型是dict，键为层号，值为 List 类型，存放着该层的个体
        """
        F = defaultdict(list)

        for p in P:
            p.dominate_set = []
            p.dominated_num = 0
            for q in P:
                if p < q:  # if p dominate q
                    p.dominate_set.append(q)  # Add q to the set of solutions dominated by p
                elif q < p:
                    p.dominated_num += 1  # Increment the domination counter of p
            if p.dominated_num == 0:
                p.rank = 1
                F[1].append(p)

        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.dominate_set:
                    q.dominated_num -= 1
                    if q.dominated_num == 0:
                        q.rank = i + 1
                        Q.append(q)
            i = i + 1
            F[i] = Q

        return F

    def crowding_distance_assignment(self, L: list):
        """ 传进来的参数应该是L = F(i)，类型是List"""
        l = len(L)  # number of solution in F
        if l < 3:
            return
        for i in range(l):
            L[i].distance = 0  # initialize distance

        for m in range(len(NSGAIndividual.fitness_calculators)):
            L.sort(key=lambda x: x.fitness[m])  # sort using each objective value
            L[0].distance = float('inf')
            L[l - 1].distance = float('inf')  # so that boundary points are always selected

            # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
            f_max = L[l - 1].fitness[m]
            f_min = L[0].fitness[m]

            # for i in range(1, l - 1):  # for all other points
            #     L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)

            if f_max != f_min:
                for i in range(1, l - 1):  # for all other points
                    L[i].distance = L[i].distance + (L[i + 1].fitness[m] - L[i - 1].fitness[m]) / (f_max - f_min)

    def binary_tournament(self, ind1, ind2):
        """
        二元锦标赛
        :param ind1:个体1号
        :param ind2: 个体2号
        :return:返回较优的个体
        """
        if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
            return ind1 if ind1.rank < ind2.rank else ind2
        elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
            return ind1 if ind1.distance > ind2.distance else ind2
        else:  # 如果rank和拥挤度都相同，返回任意一个都可以
            return ind1

    def get_best_in_solution(self, fraction=None, selected=None):
        # 比例优化空间
        if fraction is None:
            step = 1.0 / (self.solution_num - 1)
            first = random.randint(0, self.solution_num-1)
            fraction = torch.tensor([max(first*step, 1e-6), max(1.0-first*step, 1e-6)])
            print('fraction:', fraction)
        else:
            fraction = torch.tensor(fraction)
        fitness_front = [p.fitness for p in self.best_solution]
        front_tensor = torch.tensor(fitness_front)
        # greedy_best = torch.tensor(self.greedy_best_fitness_points)
        best_point = torch.min(front_tensor, dim=0).values
        worst_point = torch.max(front_tensor, dim=0).values
        scores = (front_tensor - best_point) / (worst_point - best_point+1e-8)

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
            if index < len(scores):
                best = sorted_list[index]
            else:
                best = torch.argmin(scores)
        return best

    def get_multi_best_solution(self):
        best_list = []
        step = 1.0 / (self.solution_num - 1)
        for i in range(self.solution_num):
            fraction = [max(i * step, 0.00001), max(1.0 - i * step, 0.00001)]
            best_list.append(self.get_best_in_solution(fraction, best_list))
        return best_list


    def make_new_pop(self, P):
        """
            use select,crossover and mutation to create a new population Q
            :param P: 父代种群
            :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
            :param bound_min: 定义域下限
            :param bound_max: 定义域上限
            :param objective_fun: 目标函数
            :return Q : 子代种群
            """
        popnum = len(P)
        Q = []
        # binary tournament selection
        for i in range(int(popnum / 2)):
            # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent1 = self.binary_tournament(P[i], P[j])

            # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = self.binary_tournament(P[i], P[j])

            while parent1.gene == parent2.gene:  # 如果选择到的两个父代完全一样，则重选另一个
                i = random.randint(0, popnum - 1)
                j = random.randint(0, popnum - 1)
                parent2 = self.binary_tournament(P[i], P[j])

            # parent1 和 parent1 进行交叉，变异 产生 2 个子代
            Two_offspring = parent1.crossover(parent2)
            child_3 = parent1.mutation()
            child_4 = parent2.mutation()
            # 产生的子代进入子代种群
            Q.append(Two_offspring[0])
            Q.append(Two_offspring[1])
            Q.append(child_3)
            Q.append(child_4)
        return Q

    def solve(self, iter=50):
        Q = self.make_new_pop(self.best_solution)

        P_t = self.best_solution
        Q_t = Q
        for cur_iter in range(iter):
            print("Iter:", cur_iter)
            R_t = P_t+Q_t
            F = self.fast_non_dominated_sort(R_t)
            P_n = []
            i = 1
            while len(P_n) + len(F[i]) < self.population_num:  # until the parent population is filled

                P_n = P_n + F[i]  # include ith non dominated front in the parent pop
                i = i + 1  # check the next front for inclusion
            self.crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
            F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
            P_n = P_n + F[i][len(F[i]) - self.population_num + len(P_n):]
            Q_n = self.make_new_pop(P_n)  # use selection,crossover and mutation to create a new population Q_n

            # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
            P_t = P_n
            Q_t = Q_n
            self.best_solution = P_t
            fitness_front = [p.fitness for p in self.best_solution]
            best = self.get_best_in_solution()
            if cur_iter % 10 == 0:
                plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, title="Iter_pareto_{}_{}".format(self.fraction, cur_iter),
                                 important_points=[self.best_solution[best].fitness], folder_name=self.output_folder)
        # 对帕累托前沿的点进行评分并选择最优解
        best_list = self.get_multi_best_solution()
        fitness_front = [p.fitness for p in self.best_solution]
        best_results = [list(self.best_solution[b].gene) for b in best_list]
        best_fitness_list = [self.best_solution[b].fitness for b in best_list]
        print("best fitness: ", best_fitness_list)
        plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, important_points=best_fitness_list,
                         title="final_pareto_{}".format(self.fraction), folder_name=self.output_folder)
        print("fitness front: ", len(fitness_front))
        with open(os.path.join(self.output_folder, 'best_solution.pkl'), 'wb') as f:
            pickle.dump(self.best_solution, f)
        return best_results, best_fitness_list, fitness_front


class NSGA2(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="Info",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        # selection_choices = ["LeastConfidence",
        #                      "Entropy",
        #                      "Confidence",
        #                      "Margin",
        #                      "Info"]
        # if selection_method not in selection_choices:
        #     raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = "Info"

        self.epochs = epochs
        self.balance = balance

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

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

                for i, (inputs, labels) in enumerate(data_loader):
                    outputs = self.model(inputs.to(self.args.device))
                    # matrix.append(outputs)
                    matrix.append(self.model.embedding_recorder.embedding)
                    row_indices = torch.arange(outputs.size(0))
                    if self.selection_method == "LeastConfidence":
                        scores = np.append(scores, outputs[[row_indices, labels]].cpu().numpy())
                    elif self.selection_method == "Entropy":
                        preds = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                    elif self.selection_method == "Info":
                        preds = torch.nn.functional.softmax(outputs, dim=1)
                        preds = preds[[row_indices, labels]].cpu().numpy()
                        scores = np.append(scores, np.log(preds + 1e-6))
                    elif self.selection_method == "Confidence":
                        preds = torch.nn.functional.softmax(outputs, dim=1)
                        preds = preds[[row_indices, labels]].cpu().numpy()
                        scores = np.append(scores, preds)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0), scores

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        if self.balance:
            selection_results = [np.array([], dtype=np.int64) for i in range(self.args.solution_num)]
            scores = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                if len(class_index) == 0:
                    continue
                test_data_folder = 'nsga_data/{}_{}/iter_{}/label_{}'.format(self.args.dataset, self.fraction, self.args.iter, c)
                os.makedirs(test_data_folder, exist_ok=True)

                features_matrix, confidence = self.construct_matrix(class_index)
                size = round(len(class_index) * self.fraction)
                time1 = time.time()
                # fitness_calculators = [RepresentativenessCalculator(features_matrix, size, device='cuda'),
                #                        DiversityCalculator(features_matrix, size, device='cuda')]
                fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
                                       InfoCalculator(features_matrix, confidence, size, device='cuda')]
                # fitness_calculators = [KLCalculator(importance, size, device='cuda'),
                #                        DiversityCalculator(features_matrix, size, device='cuda')]
                # fitness_calculators = [MMDCalculator(features_matrix, size, device='cuda'),
                #                        DiversityCalculator(features_matrix, size, device='cuda')]
                # fitness_calculators = [UniquenessCalculator(importance, size, device='cuda'),
                #                        MMDCalculator(features_matrix, size, device='cuda')]
                # fitness_calculators = [UniquenessCalculator(confidence, size, device='cuda'),
                #                        DiversityCalculator(features_matrix, size, device='cuda')]
                solver = NSGA2Alg(fitness_calculators=fitness_calculators, total_gene_num=len(class_index), budget=size,
                               device='cuda',
                               population_num=self.args.population, output_folder=test_data_folder, solution_num=self.args.solution_num)
                best_list, best_fitness, fitness_front = solver.solve(iter=self.args.iter)
                time2 = time.time()
                print("heuristic time: ", time2 - time1)
                for i in range(len(best_list)):
                    best_result = class_index[np.array(list(best_list[i]))]
                    selection_results[i] = np.append(selection_results[i], best_result)


                best_file_path = os.path.join(test_data_folder, 'best_multi.npy')
                np.save(best_file_path, np.array(best_list))
                np.save(os.path.join(test_data_folder, 'front_multi.npy'), np.array(fitness_front))
                torch.save(features_matrix,
                           os.path.join(test_data_folder, 'features_matrix.pth'))
                torch.save(confidence, os.path.join(test_data_folder, 'importance.pth'))
        else:
            selection_results = None
            # scores = self.rank_uncertainty()
            # selection_result = np.argsort(scores)[:self.coreset_size]
        test_data_folder = 'nsga_data/{}_{}/iter_{}/final'.format(self.args.dataset, self.fraction,
                                                                         self.args.iter)
        os.makedirs(test_data_folder, exist_ok=True)
        best_file_path = os.path.join(test_data_folder, 'best_multi.npy')
        np.save(best_file_path, selection_results)

        return [{'indices': result} for result in selection_results]

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
