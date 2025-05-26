import sys

from WorkBook import WorkBook
from continual_learning import continual_learning
from continual_learning_single_objective import continual_learning_for_single_objective

running_args = {
    "--dataset": "SubCIFAR100",
    "--model": "ResNet18",
    "--selection": "MicroSearch",
    # "--num_exp": 20,
    "--num_exp": 1,
    "--num_eval": 1,
    # "--epochs": 2,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.1,
    "--workers": 4,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 256,
    "--selection_batch": 256,
    "--test_interval": 1,
    # "--selection_epochs": 1,
    "--selection_epochs": 25,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.002,
    "--selection_test_interval": 1,
    "--balance": True,
    "--uncertainty": "LeastConfidence",
    "--solution_num": 5,
    "--population": 10,
    "--step_rate": 0.1,
    "--iter": 20,
    "--divide_step": 5,
    "--recall_epochs": 50
}

if __name__ == '__main__':
    wb = WorkBook(running_args["--num_exp"])

    origin_argv = sys.argv
    print(sys.argv)
    # run
    for key, value in running_args.items():
        sys.argv.append(key)
        sys.argv.append(str(value))
    print(sys.argv)
    continual_learning(wb)
    # continual_learning_for_single_objective(wb)

    wb.append('备注', 0, "MicroSearch, fraction: 0.1, model: ResNet18, dataset: CIFAR100, MicroSearch, last_layer, Info, ratio: 1.0, cosine, population: 10, iter: 20, step_rate: 0.1, mmd: 0.003")
    wb.to_excel('./excel/data_cl_MicroSearch_10_micro.xlsx')

    print("end")
