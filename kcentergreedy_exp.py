import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model

# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "kCenterGreedy",
#     # "--num_exp": 5,
#     "--num_exp": 3,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.01,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     # "--selection_epochs": 1,
#     "--selection_epochs": 25,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

# running_args = {
#     "--dataset": "CIFAR10",
#     "--model": "ResNet18",
#     "--selection": "kCenterGreedy",
#     "--num_exp": 5,
#     # "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.3,
#     "--workers": 4,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     "--selection_epochs": 25,
#     # "--selection_epochs": 1,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

# running_args = {
#     "--dataset": "CIFAR100",
#     "--model": "ResNet18",
#     "--selection": "kCenterGreedy",
#     # "--num_exp": 20,
#     "--num_exp": 5,
#     "--num_eval": 1,
#     # "--epochs": 2,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.9,
#     "--workers": 4,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     "--selection_epochs": 25,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--balance": True
# }

# running_args = {
#     "--dataset": "SST5",
#     "--model": "TextCNN",
#     "--selection": "kCenterGreedy",
#     "--num_exp": 20,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.9,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     "--selection_epochs": 25,
#     # "--selection_epochs": 2,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--balance": True
# }
#
# running_args = {
#     "--dataset": "AGNews",
#     "--model": "TextCNN",
#     "--selection": "kCenterGreedy",
#     "--num_exp": 4,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 100,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.5,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.001,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     "--selection_epochs": 12,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.001,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

running_args = {
    "--dataset": "UrbanSound8K",
    "--model": "TDNN",
    "--selection": "kCenterGreedy",
    "--num_exp": 5,
    # "--num_exp": 1,
    "--num_eval": 1,
    # "--epochs": 10,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.7,
    "--workers": 8,
    "--optimizer": "Adam",
    "--lr": 0.001,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 64,
    "--selection_batch": 64,
    "--test_interval": 1,
    "--selection_epochs": 25,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.001,
    "--selection_test_interval": 1,
    "--uncertainty": "LeastConfidence",
    "--balance": True,
}

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
    # wb.append('备注', 0, "kCenterGreedy, fraction: 0.01, model: LeNet, dataset: MNIST")
    # wb.to_excel('./excel/data_kCenterGreedy_01.xlsx')
    # wb.append('备注', 0, "kCenterGreedy, fraction: 0.9, model: ResNet18, dataset: CIFAR100, batch:256, worker:4, last_layer")
    # wb.append('备注', 0, "kCenterGreedy, fraction: 0.3 model: ResNet18, dataset: CIFAR10, worker:4, batch: 256, "
    #                    "使用last_layer作为特征矩阵")
    # wb.append('备注', 0, "kCenterGreedy, fraction: 0.9, model: TextCNN, dataset: SST5, batch:256, worker:8, last_layer")
    # wb.append('备注', 0, "kCenterGreedy, fraction: 0.5, model: TextCNN, dataset: AGNews, batch:256, worker:8, last_layer")
    #
    # wb.to_excel('./excel/data_kCenterGreedy_50.xlsx')
    wb.append('备注', 0, "kCenterGreedy, fraction: 0.7, model: TDNN, dataset: UrbanSound8K, batch:64, worker:8, last_layer")

    wb.to_excel('./excel/data_kCenterGreedy_70.xlsx')
    print("end")
