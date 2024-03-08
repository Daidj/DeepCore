from .earlytrain import EarlyTrain
import torch
import numpy as np


class UncertaintyNew(EarlyTrain):
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
        self.selected_scores = np.array([])

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        scores = self.rank_uncertainty().reshape(1, -1)
        if len(self.selected_scores) == 0:
            self.selected_scores = np.array(scores)
        else:
            self.selected_scores = np.row_stack((self.selected_scores, scores))

        # scores = self.rank_uncertainty().reshape(1, -1)

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            mean_scores = self.selected_scores.mean(axis=0)
            fraction = self.get_percent(mean_scores)
            scores = []
            # scores = self.selected_scores.mean(axis=0)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                scores.append(mean_scores[class_index])
                coreset_size = int(fraction * len(class_index))
                # sorted_scores = np.sort(scores[-1])
                # X = [i for i in range(len(scores[-1]))]
                # Y = sorted_scores
                # plt.plot(X, Y)
                # plt.show()
                # self.get_best_percent((scores[-1]))
                selection_result = np.append(selection_result, class_index[np.argsort(scores[-1])[:coreset_size]])

        else:
            scores = self.selected_scores.mean(axis=0)
            coreset_size = int(self.get_percent(scores) * self.n_train)
            print("coreset size", coreset_size)
            selection_result = np.argsort(scores)[::-1][:coreset_size]
        return {"indices": selection_result, "scores": scores}

    def get_percent(self, scores, percentile=0.99):
        scores = np.sort(scores)
        size = len(scores)
        percent = 0.0
        scores_sum_edge = scores.sum()*percentile
        step_size = 2
        while True:
            step = (1 - percent)/step_size
            new_chosen = int((percent+step) * size)
            current_sum = scores[size-new_chosen:].sum()
            if current_sum < scores_sum_edge:
                percent = percent + step
            else:
                step_size = step_size * 2
            if step < 0.01:
                break
        print("final percent: ", percent)
        return percent

    def get_best_percent(self, scores, points_num = 5):
        '''
            scores: 样本质量分数，分数越高质量越高
        '''
        # scores = np.sort(scores)
        # scores = 1 - scores
        start_percent = 0.0
        end_percent = 1.0
        # while True:
        #     step = (end_percent - start_percent)/(points_num-1)
        #     quality = np.empty(points_num-1)
        #     points_index = np.empty(points_num, dtype=np.int64)
        #     points_values_percentile = np.empty(points_num)
        #
        #     for i in range(points_num):
        #         points_values_percentile[i] = i * step + start_percent
        #         points_index[i] = int(points_values_percentile[i] * len(scores))
        #         if i > 0:
        #             quality[i-1] = scores[points_index[i-1]:points_index[i]].sum()
        #     max_index = np.argmax(quality)
        #     min_index = np.argmin(quality)
        #     start_percent = points_values_percentile[min_index+1]
        #     end_percent = points_values_percentile[max_index]
        #     if step < 0.01:
        #         break
        #     else:
        #         print("loop")

        step = (end_percent - start_percent) / points_num
        points = np.empty(points_num)
        # points_index = np.empty(points_num, dtype=np.int64)
        points_values_percentile = np.empty(points_num)
        for i in range(points_num):
            points_values_percentile[i] = (i + 1) * step + start_percent
        step_size = np.empty(points_num - 1)
        while True:

            for i in range(points_num):
                # points_index[i] = int(points_values_percentile[i] * len(scores))
                points[i] = self.get_percent(scores, points_values_percentile[i])
                if i > 0:
                    step_size[i - 1] = points[i] - points[i - 1]

            step = step/2
            min_index = np.argmin(step_size)
            max_index = np.argmax(step_size)
            points_values_percentile[max_index] = points_values_percentile[max_index] + step
            points_values_percentile[min_index+1] = points_values_percentile[min_index+1] + step
            # for i in range(points_num-1):
            #     if points[i + 1] - points[i] < step:
            #         min_index = i
            #     else:
            #         break

            # start_percent = points_values_percentile[min_index+1]
            # end_percent = points_values_percentile[max_index]
            if step < 0.001:
                break
            else:
                print("loop")


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
                    outputs = self.model(input.to(self.args.device))
                    samples_scores = outputs.max(axis=1).values.cpu().numpy()
                    scores = np.append(scores, samples_scores)
                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    scores = np.append(scores, (np.log(preds + 1e-6) * preds).sum(axis=1))
                elif self.selection_method == "Confidence":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1).cpu().numpy()
                    max_preds = preds.max(axis=1)
                    # print(max_preds)
                    # print(max_preds.shape)
                    # scores = np.append(scores, max_preds)
                    scores = np.append(scores, 1 - max_preds)
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
