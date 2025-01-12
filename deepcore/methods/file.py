import os
import random

import numpy as np
import pickle

from .coresetmethod import CoresetMethod


class File(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, **kwargs)
        self.balance = balance
        self.root = 'process_data_{}'.format(round(self.args.fraction*100))

    def finish_run(self):
        if self.balance:
            selection_results = [np.array([], dtype=np.int64) for i in range(self.args.solution_num)]
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                file = os.path.join(self.root, 'iter_{}_label_{}/best_results'.format(self.args.iter, c))
                with open(file, 'rb') as f:
                    best_results = pickle.load(f)
                for i in range(len(best_results)):
                    best_result = class_index[np.array(list(best_results[i]))]
                    selection_results[i] = np.append(selection_results[i], best_result)
        else:
            selection_result = np.array(random.sample([i for i in range(self.n_train)], self.coreset_size))
        return [{'indices': result} for result in selection_results]

    def select(self, **kwargs):
        selection_result = self.finish_run()
        return selection_result
