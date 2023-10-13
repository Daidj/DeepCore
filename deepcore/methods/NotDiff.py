from .earlytrain import EarlyTrain
import torch, time
from torch import nn
import numpy as np

from .. import nets


class NotDiff(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
                         dst_test=dst_test)

        self.candidate_set = set()
        self.train_error_set = set()
        self.valid_error_set = set()
        self.not_diff = 0
        self.balance = balance

    def get_hms(self, seconds):
        # Format time for printing purposes

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    def train_valid_model(self, epoch, list_of_train_idx, **kwargs):
        """ Train valid model for one epoch """

        self.valid_model.train()

        print('\n=> Training Epoch(Valid Model) #%d' % epoch)
        print(list_of_train_idx)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args.workers, pin_memory=True)

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.valid_model_optimizer.zero_grad()
            outputs = self.valid_model(inputs)
            loss = self.criterion(outputs, targets)
            # batch_inds = trainset_permutation_inds[i]
            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            # self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.valid_model_optimizer.step()
        return

    def run_valid_model(self):
        if len(self.candidate_set) == 0:
            return

        list_of_train_idx = np.array(list(self.candidate_set))
        print(list_of_train_idx)
        self.train_valid_model(0, list_of_train_idx)
            # if self.dst_test is not None and self.args.selection_test_interval > 0 and (
            #         epoch + 1) % self.args.selection_test_interval == 0:
            #     self.test(epoch)

    def before_train(self):
        self.train_loss = 0.
        self.correct = 0.
        self.total = 0.

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        # if epoch == 0:
        #     return
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)

            cur_acc = (predicted == targets).clone().detach().requires_grad_(False).type(torch.float32)
            # 添加进候选集
            error_index = torch.tensor(batch_inds)[(cur_acc==0).nonzero()].reshape(-1)
            self.candidate_set = self.candidate_set.union(set(error_index.numpy().tolist()))
            self.valid_error_set = self.valid_error_set.union(set(error_index.numpy().tolist()))
            print("candidate set len: ", len(self.candidate_set))
            # self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds] - cur_acc) > 0.01]] += 1.
            # self.last_acc[batch_inds] = cur_acc

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        self.train_loss += loss.item()
        self.total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        self.correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item(),
                100. * self.correct.item() / self.total))

    def before_epoch(self):
        self.start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.start_time
        self.elapsed_time += epoch_time

        print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(self.elapsed_time)))
        self.train_error_set = self.valid_error_set
        self.valid_error_set = set()
        self.run_valid_model()
        print("train len: ", len(self.train_error_set))
        print("valid len: ", len(self.valid_error_set))
        union = self.train_error_set.union(self.valid_error_set)
        inter = self.train_error_set.intersection(self.valid_error_set)
        diff = union.symmetric_difference(inter)
        print("diff: ", len(diff))
        if len(diff) < 5:
            self.not_diff += 1


    def before_run(self):
        self.elapsed_time = 0
        # self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)
        # self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

        # 配置验证模型
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Setup model and loss
        self.valid_model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.valid_model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            self.valid_model = nets.nets_utils.MyDataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.valid_model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                   momentum=self.args.selection_momentum,
                                                   weight_decay=self.args.selection_weight_decay,
                                                   nesterov=self.args.selection_nesterov)
        elif self.args.selection_optimizer == "Adam":
            self.valid_model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                    weight_decay=self.args.selection_weight_decay)
        else:
            self.valid_model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                                       lr=self.args.selection_lr,
                                                                                       momentum=self.args.selection_momentum,
                                                                                       weight_decay=self.args.selection_weight_decay,
                                                                                       nesterov=self.args.selection_nesterov)

    def finish_run(self):
        if self.not_diff > 4:
            print("Not Diff")

    def select(self, **kwargs):
        self.run()
        examples = np.array(list(self.candidate_set), dtype=np.int64)
        # if not self.balance:
        #     top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]
        # else:
        #     top_examples = np.array([], dtype=np.int64)
        #     for c in range(self.num_classes):
        #         c_indx = self.train_indx[self.dst_train.targets == c]
        #         budget = round(self.fraction * len(c_indx))
        #         top_examples = np.append(top_examples,
        #                                  c_indx[
        #                                      np.argsort(self.forgetting_events[c_indx].cpu().numpy())[::-1][:budget]])
        print(examples.shape)
        return {"indices": examples}

    def need_stop(self):
        if self.not_diff > 4:

            return True
        else:
            return False
