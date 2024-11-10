from mmd_algorithm import MMD
from . import CoresetMethod
from .earlytrain import EarlyTrain
import torch
import numpy as np
import matplotlib.pyplot as plt


class MMDDistance(EarlyTrain):
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
        self.mmd_calculators = []

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
            selection_distance = np.array([], dtype=np.float64)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                features_matrix, importance = self.construct_matrix(class_index)
                if c <= len(self.mmd_calculators):
                    calculator = MMD(features_matrix)
                    self.mmd_calculators.append(calculator)
                else:
                    calculator = self.mmd_calculators[c]
                subset_size = round(self.fraction * len(class_index))
                # best_result, dis = self.get_best_subset_random(calculator, subset_size, class_index, 5)
                best_set, dis = self.get_best_subset_2(calculator, subset_size)

                best_result = class_index[best_set]
                print('best: ', best_result)
                print('distance: ', dis)
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
        selection_result = self.run()
        return selection_result

    '''随机生成'''

    def get_best_subset(self, calculator: MMD, subset_size, full_set, iter=100):
        best_set = None
        best_distance = None
        for i in range(iter):
            selected_set = np.random.choice(np.arange(len(full_set)), subset_size, replace=False)
            distance = calculator.mmd_for_data_set(torch.tensor(selected_set))
            if best_distance is None or best_distance > distance:
                best_distance = distance
                best_set = selected_set
        return full_set[best_set], best_distance

    def get_worst_subset(self, calculator: MMD, subset_size, full_set, iter=100):
        worst_set = None
        worst_distance = None
        for i in range(iter):
            selected_set = np.random.choice(np.arange(len(full_set)), subset_size, replace=False)
            distance = calculator.mmd_for_data_set(torch.tensor(selected_set))
            if worst_distance is None or worst_distance < distance:
                worst_distance = distance
                worst_set = selected_set
        return full_set[worst_set], worst_distance

    def get_best_subset_2(self, calculator: MMD, subset_size):
        best_set = calculator.get_min_distance_index(subset_size)
        # best_set = np.sort(best_set)
        best_distance = calculator.mmd_for_data_set(torch.tensor(best_set))
        return best_set, best_distance

    def get_best_subset_random(self, calculator: MMD, subset_size, full_set, iter=50):
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
