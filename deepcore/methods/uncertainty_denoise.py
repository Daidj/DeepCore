import os
import time
import pickle
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


class UncertaintyDenoise(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="Confidence",
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

    def write_file(self, class_index, scores, id, name='class'):
        folder_name = 'test_data'
        os.makedirs(folder_name, exist_ok=True)

        # 构造文件路径
        file_path = os.path.join(folder_name, '{}_{}.pkl'.format(name, id))
        my_array = class_index[np.argsort(scores[-1])]
        # 将数组保存到文件
        with open(file_path, 'wb') as file:
            pickle.dump(my_array, file)

        # 重新从文件中读取数组
        # with open(file_path, 'rb') as file:
        #     loaded_array = pickle.load(file)

        # 打印读取的数组
        # print(loaded_array)

    def finish_run(self):
        start_time = time.time()
        mmd_time = 0.0
        selection_distance = None
        noise_index = self.get_noise()
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            selection_distance = np.array([], dtype=np.float64)
            scores = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]


                scores.append(self.rank_uncertainty(class_index))
                start_index = 0
                size = round(len(class_index) * self.fraction)

                noise_index_for_class = np.where(np.isin(class_index, noise_index))[0]
                scores[-1][noise_index_for_class] = scores[-1][noise_index_for_class] * 2
                filtered_index = class_index[np.argsort(scores[-1])]
                # selected_index = class_index[np.argsort(scores[-1])]
                # filtered_index = selected_index[~np.isin(selected_index, noise_index)]

                # self.write_file(class_index, scores, c, name="CIFAR10-fraction0_5")
                best_result = filtered_index[start_index:start_index + size]
                kl = 0.0
                # kl = self.kl_divergence(scores[c], scores[c][selected_index])
                selection_distance = np.append(selection_distance, kl)
                selection_result = np.append(selection_result, best_result)


        else:
            scores = self.rank_uncertainty()
            selection_result = np.argsort(scores)[:self.coreset_size]
        end_time = time.time()
        mmd_time += (end_time - start_time)

        return {"indices": selection_result, "scores": scores, "mmd_distance": selection_distance, "mmd_time": mmd_time}

    def get_noise(self, index=None):
        scores = None
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)

            batch_num = len(train_loader)

            for i, (input, _) in enumerate(train_loader):
                preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                # max_preds = preds.max(axis=1)
                # print(max_preds)
                # print(max_preds.shape)

                scores = np.concatenate((scores, preds), axis=0) if scores is not None else preds

        # 提取图像数据和标签
        labels = np.reshape(self.dst_train.targets.numpy() if index is None else self.dst_train.targets.numpy()[index], -1)

        # 获取噪声索引
        noise_idx = find_label_issues(
            labels=labels,  # 真实标签
            pred_probs=scores,  # 分类器的概率输出
            filter_by='prune_by_noise_rate',  # 根据类别修剪噪声
        )
        noise_idx = np.where(np.array(noise_idx))[0]
        # folder_name = 'test_data'
        # os.makedirs(folder_name, exist_ok=True)
        # file_path = os.path.join(folder_name, 'error_CIFAR10.pkl')
        # with open(file_path, 'wb') as file:
        #     pickle.dump(noise_idx, file)
        return noise_idx

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch,
                num_workers=self.args.workers)

            scores = np.array([])
            batch_num = len(train_loader)

            for i, (input, _) in enumerate(train_loader):
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":
                    scores = np.append(scores, self.model(input.to(self.args.device)).max(axis=1).values.cpu().numpy())
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == "Confidence":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    max_preds = preds.max(axis=1)
                    # print(max_preds)
                    # print(max_preds.shape)
                    scores = np.append(scores, max_preds)
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
