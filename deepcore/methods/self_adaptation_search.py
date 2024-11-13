import math
import random

from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist, euclidean_dist_for_batch, MMD, micro_search
from ..nets.nets_utils import MyDataParallel

def self_adaptation_search(matrix, confidence, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
        matrix = matrix.to(device)
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    if type(confidence) == np.ndarray:
        confidence = torch.from_numpy(confidence).requires_grad_(False).to(device)
    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num
    budget = sample_num
    init_num = budget
    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    selected = set()
    unselected = set(torch.arange(sample_num).numpy())
    distance = euclidean_dist_for_batch(matrix, matrix)
    confidence = confidence.cpu()
    confidence = (confidence - confidence.min() + 1e-3) / (confidence.max() - confidence.min())
    # middle = (1-fraction)/2
    # confidence = torch.abs(confidence-middle)/(0.5+0.5*fraction)

    # confidence = torch.abs(confidence-(1-fraction))/max(fraction, (1-fraction))
    similarity_redundancy_ratio = 1.0

    subset, mmd_search_num = micro_search.first_stage_search(matrix, init_num)
    init_num -= mmd_search_num
    selected = unselected
    second_stage_search = set()
    second_stage_search.update(selected)
    second_stage_search.difference_update(subset)
    unselected = set()

    print('mmd search end: {}/{}({})'.format(len(selected), sample_num, len(selected)/sample_num))
    print('uncertainty serach: {}/{}({})'.format(init_num, sample_num, init_num/sample_num))
    step = 1

    finish = False
    while not finish:
        selected_tensor = torch.tensor(list(selected))
        unselected_tensor = torch.tensor(list(unselected))
        selected_dis = distance[selected_tensor, :][:, selected_tensor]
        selected_min_dis = torch.kthvalue(selected_dis, 2, dim=1).values
        selected_bak_dis = torch.kthvalue(selected_dis, 3, dim=1).values

        result_tensor = torch.where(selected_dis != selected_min_dis, torch.tensor(0.0),
                                    selected_bak_dis - selected_dis)
        other_dis = torch.sum(result_tensor, dim=1)
        redundancy_info = (selected_min_dis - other_dis)
        uncertainty = confidence[selected_tensor]

        scores = uncertainty - similarity_redundancy_ratio * redundancy_info
        current_score = torch.sum(scores).item()
        sorted_tensor, sorted_indices = torch.sort(scores, descending=True)
        l = list(selected)
        for cur in range(len(sorted_indices)):
            if l[sorted_indices[cur]] in second_stage_search:
                if sorted_tensor[cur] > 1.99:
                    selected.remove(l[sorted_indices[cur]])
                    second_stage_search.remove(l[sorted_indices[cur]])
                    unselected.add(l[sorted_indices[cur]])

                else:
                    finish = True
                break
        print('score: {}'.format(current_score))


    assert len(selected) == budget
    return index[list(selected)], mmd_search_num


class SelfAdaptationSearch(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=0,
                 specific_model=None, balance: bool = False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)

        self.min_distances = None
        self.selection_method = "Info"

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
        self.balance = balance

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
                        # max_preds = preds.max(axis=1)
                        preds = preds[[row_indices, labels]].cpu().numpy()
                        # print(max_preds)
                        # print(max_preds.shape)
                        scores = np.append(scores, preds)

        self.model.no_grad = False
        return torch.cat(matrix, dim=0), scores

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def select(self, **kwargs):
        self.run()
        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            entroy_list = []
            class_index_list = []
            class_num = []
            features_matrix_list = []
            confidence_list = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                class_num.append(len(class_index))
                features_matrix, confidence = self.construct_matrix(class_index)
                class_index_list.append(class_index)
                entroy_list.append(confidence.sum())
                features_matrix_list.append(features_matrix)
                confidence_list.append(confidence)
            budget_list = -torch.tensor(entroy_list)
            budget_list = ((budget_list/budget_list.sum())*self.fraction*self.n_train).numpy()
            balance = False
            not_full_index = set([i for i in range(len(class_num))])
            while not balance:
                balance = True

                for i in range(len(class_num)):
                    if budget_list[i] > class_num[i]:
                        not_full_index.remove(i)
                        rest = (budget_list[i]-class_num[i])/(len(not_full_index))
                        budget_list[list(not_full_index)] = budget_list[list(not_full_index)] + rest
                        budget_list[i] = class_num[i]
                        balance = False

            mmd_search_total = 0
            for c in range(self.args.num_classes):
                class_index = class_index_list[c]
                features_matrix = features_matrix_list[c]
                confidence = confidence_list[c]
                # print('imbalance')
                # budget = round(budget_list[c])
                print('balance')
                budget = round(self.fraction*len(class_index))
                subset, mmd_search_num = self_adaptation_search(features_matrix, confidence,
                                                                               budget=budget,
                                                                               metric=self.metric,
                                                                               device=self.args.device,
                                                                               random_seed=self.random_seed,
                                                                               index=class_index,
                                                                               already_selected=self.already_selected[
                                                                                   np.in1d(self.already_selected,
                                                                                           class_index)],
                                                                               print_freq=self.args.print_freq)
                selection_result = np.append(selection_result, subset)
                mmd_search_total += mmd_search_num
                features_matrix_list[c] = None
        else:
            pass
            # matrix = self.construct_matrix()
            # del self.model_optimizer
            # del self.model
            # selection_result = k_center_greedy(matrix, budget=self.coreset_size,
            #                                    metric=self.metric, device=self.args.device,
            #                                    random_seed=self.random_seed,
            #                                    already_selected=self.already_selected, print_freq=self.args.print_freq)

        return {"indices": selection_result, "mmd_distance": mmd_search_total}
