import random

import torch
import numpy as np

from mmd_algorithm import MMD


def first_stage_search(matrix, budget: int, metric=None, device='cuda', search=True):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
        matrix = matrix.to(device)
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num
    init_num = budget

    selected = set()
    unselected = set(torch.arange(sample_num).numpy())
    stage_budget = min(round(0.01*sample_num), budget)
    step = max(1, round(stage_budget * 0.1))

    # second_stage_budget = max(0, budget-first_stage_budget)
    calculator = MMD(matrix, 'cuda')
    min_mmd_distance = 0.003
    mmd_distance = 1.0
    while init_num > 0:
        if mmd_distance < min_mmd_distance:
            break
        # print('mmd distance: {}'.format(mmd_distance))
        # print('mmd serach: {}'.format(stage_budget))

        current_stage_budget = min(stage_budget, init_num)
        while current_stage_budget > 0:
            selected_num = min(current_stage_budget, step)
            if len(selected) == 0:
                # selected_num = max(5, selected_num)
                selected = set(random.sample(list(unselected), selected_num))
                unselected.difference_update(selected)
            else:

                # mmd_scores = calculator.get_selected_scores(selected, unselected)
                # mmd_scores = (mmd_scores - mmd_scores.min()) / (mmd_scores.max() - mmd_scores.min())
                #
                # _, indices = torch.topk(mmd_scores, k=step)
                # l = list(selected)
                # remove = set([l[i] for i in indices])
                # selected.difference_update(remove)
                # unselected.update(remove)

                mmd_scores = calculator.get_unselected_scores(selected, unselected)
                scores = (mmd_scores - mmd_scores.min()) / (mmd_scores.max() - mmd_scores.min())
                _, indices = torch.topk(scores, k=selected_num, largest=False)
                l = list(unselected)
                new_elements = set([l[i] for i in indices])
                selected.update(new_elements)
                unselected.difference_update(selected)
            current_stage_budget -= selected_num
            init_num -= selected_num
        if search:
            have_remove = set()
            for i in range(20):
                search_step = 1
                mmd_scores = calculator.get_selected_scores(selected, unselected)
                _, indices = torch.topk(mmd_scores, k=search_step)
                l = list(selected)
                remove = set([l[i] for i in indices])
                if remove.issubset(have_remove):
                    print('early stop')
                    break
                have_remove.update(remove)
                selected.difference_update(remove)
                unselected.update(remove)

                mmd_scores = calculator.get_unselected_scores(selected, unselected)
                _, indices = torch.topk(mmd_scores, k=search_step, largest=False)
                l = list(unselected)
                new_elements = set([l[i] for i in indices])
                selected.update(new_elements)
                unselected.difference_update(selected)
                # mmd_distance = calculator.mmd_for_data_set(torch.tensor(list(selected)))
                # print(mmd_distance)
        mmd_distance = calculator.mmd_for_data_set(torch.tensor(list(selected)))
    print('mmd search end: {}/{}({})'.format(len(selected), sample_num, len(selected)/sample_num))
    print('uncertainty serach: {}/{}({})'.format(init_num, sample_num, init_num/sample_num))
    mmd_search_num = len(selected)

    return list(selected), mmd_search_num