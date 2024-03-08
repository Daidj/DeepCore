from mmd_algorithm import MMD
from . import CoresetMethod
from .earlytrain import EarlyTrain
import torch
import numpy as np
import matplotlib.pyplot as plt


class MMDDistance(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, balance=True, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, **kwargs)
        self.balance = balance
        self.mmd_calculators = []

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            selection_distance = np.array([], dtype=np.float64)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                if c <= len(self.mmd_calculators):
                    calculator = MMD(self.dst_train.data[class_index])
                    self.mmd_calculators.append(calculator)
                else:
                    calculator = self.mmd_calculators[c]
                subset_size = round(self.fraction*len(class_index))
                best_result, dis = self.get_best_subset_random(calculator, subset_size, class_index, 10)
                # best_result, dis = self.get_worst_subset(calculator, subset_size, class_index)
                selection_result = np.append(selection_result, best_result)
                selection_distance = np.append(selection_distance, dis)
        else:
            pass
            # print(self.selected_scores.shape)
            # scores = self.selected_scores.mean(axis=0)
            #
            # coreset_size = int(self.get_percent(scores) * self.n_train)
            # print("coreset size", coreset_size)
            # print("final score:", scores.shape)
            # selection_result = np.argsort(scores)[::-1][:coreset_size]
            # sorted_scores = np.sort(scores)
            # X = [i for i in range(len(scores))]
            # Y = sorted_scores
            # plt.plot(X, Y)
            # plt.show()
        return {"indices": selection_result, "mmd_distance": selection_distance}

    def select(self, **kwargs):
        selection_result = self.finish_run()
        return selection_result

    '''随机生成'''
    def get_best_subset(self, calculator:MMD, subset_size, full_set, iter = 100):
        best_set = None
        best_distance = None
        for i in range(iter):
            selected_set = np.random.choice(np.arange(len(full_set)), subset_size, replace=False)
            distance = calculator.mmd_for_data_set(torch.tensor(selected_set))
            if best_distance is None or best_distance > distance:
                best_distance = distance
                best_set = selected_set
        return full_set[best_set], best_distance

    def get_worst_subset(self, calculator:MMD, subset_size, full_set, iter = 100):
        worst_set = None
        worst_distance = None
        for i in range(iter):
            selected_set = np.random.choice(np.arange(len(full_set)), subset_size, replace=False)
            distance = calculator.mmd_for_data_set(torch.tensor(selected_set))
            if worst_distance is None or worst_distance < distance:
                worst_distance = distance
                worst_set = selected_set
        return full_set[worst_set], worst_distance

    def get_best_subset_2(self, calculator:MMD, subset_size, full_set, iter = 100):
        best_set = calculator.get_min_distance_index(subset_size)
        # best_set = np.sort(best_set)
        best_distance = calculator.mmd_for_data_set(torch.tensor(best_set))
        return full_set[best_set], best_distance

    def get_best_subset_random(self, calculator:MMD, subset_size, full_set, iter = 50):
        best_set = calculator.get_min_distance_index_with_random(subset_size)
        # best_set = np.sort(best_set)
        best_distance = calculator.mmd_for_data_set(torch.tensor(best_set))
        for i in range(iter):
            selected_set = calculator.get_min_distance_index_with_random(subset_size)
            distance = calculator.mmd_for_data_set(torch.tensor(best_set))
            if best_distance is None or best_distance > distance:
                best_distance = distance
                best_set = selected_set

        return full_set[best_set], best_distance