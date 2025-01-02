import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model
#
# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "fdmat",
#     # "--num_exp": 10,
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
#     # "--selection_epochs": 2,
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
#     "--selection": "fdmat",
#     # "--num_exp": 5,
#     "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.5,
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
#     # "--selection_epochs": 10,
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
#     "--selection": "fdmat",
#     # "--num_exp": 20,
#     "--num_exp": 1,
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

running_args = {
    "--dataset": "SST5",
    "--model": "TextCNN",
    "--selection": "fdmat",
    "--num_exp": 5,
    # "--num_exp": 1,
    "--num_eval": 1,
    # "--epochs": 10,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.1,
    "--workers": 8,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 256,
    "--selection_batch": 256,
    "--test_interval": 1,
    "--selection_epochs": 25,
    # "--selection_epochs": 2,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.002,
    "--selection_test_interval": 1,
    "--balance": True
}

# running_args = {
#     "--dataset": "UrbanSound8K",
#     "--model": "TDNN",
#     "--selection": "fdmat",
#     "--num_exp": 5,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 200,
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
#     "--train_batch": 64,
#     "--selection_batch": 64,
#     "--test_interval": 1,
#     "--selection_epochs": 25,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.001,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True,
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
    # wb.append('备注', 0, "fdmat, fraction: 0.01, model: LeNet, dataset: MNIST")
    # wb.to_excel('./excel/data_fdmat_01_mnist.xlsx')

    # wb.append('备注', 0, "fdmat, fraction: 0.5, model: ResNet18, dataset: CIFAR10")
    # wb.to_excel('./excel/data_fdmat_50.xlsx')

    # wb.append('备注', 0, "fdmat, fraction: 0.9, model: ResNet18, dataset: CIFAR100")
    # wb.to_excel('./excel/data_fdmat_90.xlsx')

    wb.append('备注', 0, "fdmat, fraction: 0.1, model: TextCNN, dataset: SST-5")
    wb.to_excel('./excel/data_fdmat_10.xlsx')

    # wb.append('备注', 0, "fdmat, fraction: 0.5, model: TDNN, dataset: UrbanSound8K")
    # wb.to_excel('./excel/data_fdmat_50.xlsx')
    print("end")
