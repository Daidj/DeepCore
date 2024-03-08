import random

import numpy as np
from .coresetmethod import CoresetMethod


class Random(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, **kwargs)
        self.balance = balance

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                selected_num = round(len(class_index)*self.fraction)
                selection_result = np.append(selection_result, np.random.choice(class_index, selected_num, replace=False))
        else:
            selection_result = np.array(random.sample([i for i in range(self.n_train)], self.coreset_size))
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.finish_run()
        return selection_result
