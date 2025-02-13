import copy
import os
import time

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from .moea_d_ldea_new import SubProblems, Individual
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


class MODED_Decompose:
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
        self.subproblems = SubProblems(target_num=self.target_num, count=population_num, device=device)
        self.best_population_for_subproblems = [None for i in range(population_num)]
        self.best_solution = []
        self.best_solution_to_subproblem = []

        Individual.fitness_calculators = fitness_calculators
        for i in [0, -1]:
            individual = Individual(total_gene_num=self.total_gene_num, gene_num=self.gene_num,
                                    target_num=self.target_num, step_rate=self.step_rate)
            individual.random_init()
            self.best_population_for_subproblems[i] = individual
        self.update_search_weight()
        self.best_population_for_subproblems[0].set_fitness()
        self.best_population_for_subproblems[-1].set_fitness()
        for i in range(1, population_num-1):
            individual = Individual(total_gene_num=self.total_gene_num, gene_num=self.gene_num,
                                    target_num=self.target_num, step_rate=self.step_rate)
            individual.random_init()
            self.best_population_for_subproblems[i] = individual

        # self.best_population_for_pareto = self.best_population_for_subproblems
        # self.best_population_for_pareto = bubble_sort(self.best_population_for_pareto)
        # sorted(self.best_population_for_pareto)


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
        self.last_best_index = int(len(self.best_solution) / 2)
        self.device = device
        self.greedy_best = []
        # for calculator in fitness_calculators:
        #     i = Individual(self.total_gene_num, self.gene_num, self.target_num)
        #     i.init(calculator.get_best())
        #     self.greedy_best.append(i)
        # self.greedy_best_fitness_points = [p.fitness for p in self.greedy_best]
        self.greedy_best_fitness_points = None
        self.output_folder = output_folder


    def update_search_weight(self):
        solution = self.best_solution
        if len(solution) == 0:
            solution = self.best_population_for_subproblems
        fitness_front = [p.origin_fitness for p in solution if p is not None]
        front_tensor = torch.tensor(fitness_front)
        best_point = torch.min(front_tensor, dim=0).values
        worst_point = torch.max(front_tensor, dim=0).values

        Individual.fitness_calculators[0].set_min_fitness(best_point[0].item())
        Individual.fitness_calculators[0].set_max_fitness(worst_point[0].item())
        Individual.fitness_calculators[1].set_min_fitness(best_point[1].item())
        Individual.fitness_calculators[1].set_max_fitness(worst_point[1].item())

    def get_best_in_solution(self, fraction=None):
        # 比例优化空间
        if fraction is None:
            step = 1.0 / (self.solution_num - 1)
            first = random.randint(0, self.solution_num-1)
            fraction = torch.tensor([first*step, 1.0-first*step])
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
        best = torch.argmin(scores)
        return best

    def get_multi_best_solution(self):
        best_list = []
        step = 1.0 / (self.solution_num - 1)
        for i in range(self.solution_num):
            fraction = [max(i * step, 0.00001), max(1.0 - i * step, 0.00001)]
            best_list.append(self.get_best_in_solution(fraction=fraction))
        return best_list

    def update_best_solution(self):

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
        print("update best solution: ", len(self.best_solution))

    def solve(self, iter=50):
        L = 10
        utility = np.ones((self.population_num, L))

        beta = min(self.target_num / 5, 0.8)
        subproblem_count = [0 for i in range(self.population_num)]
        for i in range(iter):
            print("Iter:", i)
            utility[:, i % L] = 1
            for opr in range(self.population_num):

                regions = self.subproblems.regions
                local_probability = np.array([])
                for region in regions:
                    local_probability = np.append(local_probability, 1 / self.count_set[list(region)].sum())
                selected_region = np.argmax(local_probability)
                if random.random() < beta:

                    selected_subproblem = random.choice(list(self.subproblems.regions[selected_region]))

                else:
                    if i == 0 and opr < 0.5 * L:
                        print("random subproblem")
                        selected_subproblem = random.randint(0, self.population_num - 1)
                    else:
                        utility_sum = np.sum(utility, axis=1)
                        selected_subproblem = np.argmax(utility_sum)
                # print('subprobleam: ', selected_subproblem)
                subproblem_count[selected_subproblem] += 1
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
                        self.count_set[new_population_subproblems[j]] = self.count_set[
                                                                            new_population_subproblems[j]] + 1
                    if not current > new_population[0]:
                        utility[selected_subproblem][i % L] = utility[selected_subproblem][i % L] + 1
            # if i % 5 == 4:
            #     self.update_best_solution()
            fitness_front = [p.fitness for p in self.best_solution]
            subproblems_front = [p.fitness for p in self.best_population_for_subproblems]
            best = self.get_best_in_solution()
            self.last_best_index = best
            self.update_search_weight()
            if i % 10 == 0:
                plot_nested_list(subproblems_front, diff=self.greedy_best_fitness_points,
                                 title="Iter_subproblems_{}_{}".format(self.fraction, i),
                                 important_points=[self.best_solution[best].fitness], folder_name=self.output_folder)
                plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, title="Iter_pareto_{}_{}".format(self.fraction,i),
                                 important_points=[self.best_solution[best].fitness], folder_name=self.output_folder)

        print('subprobleam: ', subproblem_count)

        self.update_best_solution()
        # 对帕累托前沿的点进行评分并选择最优解
        best_list = self.get_multi_best_solution()
        fitness_front = [p.fitness for p in self.best_solution]
        best_results = [list(self.best_solution[b].gene) for b in best_list]
        best_fitness_list = [self.best_solution[b].fitness for b in best_list]
        print("best fitness: ", best_fitness_list)
        plot_nested_list(fitness_front, diff=self.greedy_best_fitness_points, important_points=best_fitness_list,
                         title="final_pareto_{}".format(self.fraction), folder_name=self.output_folder)
        print("fitness front: ", len(fitness_front))
        return best_results, best_fitness_list, fitness_front


class MOEAD_DE(EarlyTrain):
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
                test_data_folder = 'test_data/iter_{}/decompose_{}'.format(self.args.iter, c)
                os.makedirs(test_data_folder, exist_ok=True)

                features_matrix, confidence = self.construct_matrix(class_index)
                # data = features_matrix.cpu().numpy()
                # pca = PCA(n_components=1)
                # features_transformed = pca.fit_transform(data).ravel()
                # frequency = self.error_events.cpu().numpy()[class_index]
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
                solver = MODED_Decompose(fitness_calculators=fitness_calculators, total_gene_num=len(class_index), budget=size,
                               device='cuda',
                               population_num=self.args.population, output_folder=test_data_folder, solution_num=self.args.solution_num, step_rate=self.args.step_rate)
                best_list, best_fitness, fitness_front = solver.solve(iter=self.args.iter)
                time2 = time.time()
                print("heuristic time: ", time2 - time1)
                for i in range(len(best_list)):
                    best_result = class_index[np.array(list(best_list[i]))]
                    selection_results[i] = np.append(selection_results[i], best_result)


                best_file_path = os.path.join(test_data_folder, 'best_{}_multi_{}.npy'.format(self.fraction, self.args.dataset))
                np.save(best_file_path, np.array(best_list))
                np.save(os.path.join(test_data_folder, 'front_{}_multi_{}.npy'.format(self.fraction, self.args.dataset)), np.array(fitness_front))
                torch.save(features_matrix,
                           os.path.join(test_data_folder, 'features_matrix_{}_{}.pth'.format(self.fraction, self.args.dataset)))
                torch.save(confidence, os.path.join(test_data_folder, 'importance_{}_{}.pth'.format(self.fraction, self.args.dataset)))
        else:
            selection_results = None
            # scores = self.rank_uncertainty()
            # selection_result = np.argsort(scores)[:self.coreset_size]
        test_data_folder = 'test_data/iter_{}/decompose_{}'.format(self.args.iter, self.args.dataset)
        os.makedirs(test_data_folder, exist_ok=True)
        best_file_path = os.path.join(test_data_folder, 'best_multi_{}.npy'.format(self.fraction))
        np.save(best_file_path, selection_results)

        return [{'indices': result} for result in selection_results]

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
