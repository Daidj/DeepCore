import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model

# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "Heuristic",
#     # "--num_exp": 20,
#     "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.1,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     # "--selection_epochs": 25,
#     "--selection_epochs": 1,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

running_args = {
    "--dataset": "CIFAR10",
    "--model": "ResNet18",
    "--selection": "Heuristic",
    # "--num_exp": 5,
    "--num_exp": 2,
    "--num_eval": 1,
    # "--epochs": 20,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.5,
    "--workers": 4,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 256,
    "--selection_batch": 256,
    "--test_interval": 1,
    "--selection_epochs": 25,
    # "--selection_epochs": 1,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.002,
    "--selection_test_interval": 1,
    "--uncertainty": "LeastConfidence",
    "--balance": True
}

# running_args = {
#     "--dataset": "THUCNews",
#     "--model": "TextCNN",
#     "--selection": "Glister",
#     "--num_exp": 20,
#     # "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 100,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.8,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 4096,
#     "--selection_batch": 4096,
#     "--test_interval": 1,
#     "--selection_epochs": 12,
#     # "--selection_epochs": 1,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

if __name__ == '__main__':
    # test_model()
    wb = WorkBook(running_args["--num_exp"])

    origin_argv = sys.argv
    print(sys.argv)
    # run
    for key, value in running_args.items():
        sys.argv.append(key)
        sys.argv.append(str(value))
    print(sys.argv)
    main(wb)
    # wb.append('备注', 0, "Heuristic, fraction: 0.1, model: LeNet, dataset:MNIST")
    # wb.to_excel('./excel/data_heuristic_10.xlsx')
    wb.append('备注', 0, "Heuristic, fraction: 0.5, model: ResNet18, dataset: CIFAR10")
    wb.to_excel('./excel/data_heuristic_50.xlsx')
    print("end")
