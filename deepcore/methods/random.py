import random

import numpy as np
from .coresetmethod import CoresetMethod


class Random(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, **kwargs):
        self.n_train = len(dst_train)
        self.fraction = fraction

    def select(self, **kwargs):
        select_num = int(self.n_train * self.fraction)
        print(self.fraction)
        return {"indices": np.array(random.sample([i for i in range(self.n_train)], select_num))}
