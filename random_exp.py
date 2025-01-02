import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model

# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "Random",
#     # "--num_exp": 5,
#     "--num_exp": 3,
#     "--num_eval": 1,
#     # "--epochs": 2,
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
#     "--selection_epochs": 25,
#     # "--selection_epochs": 2,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--balance": True
# }

# running_args = {
#     "--dataset": "CIFAR100",
#     "--model": "ResNet18",
#     "--selection": "Random",
#     # "--num_exp": 20,
#     "--num_exp": 5,
#     "--num_eval": 1,
#     # "--epochs": 2,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 1.0,
#     "--workers": 4,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
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
    "--dataset": "CIFAR10",
    "--model": "ResNet18",
    "--selection": "Random",
    # "--num_exp": 20,
    "--num_exp": 1,
    "--num_eval": 1,
    # "--epochs": 2,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.2,
    "--workers": 4,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 256,
    "--test_interval": 1,
    "--selection_epochs": 25,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.002,
    "--selection_test_interval": 1,
    "--balance": True
}

# running_args = {
#     "--dataset": "TinyImageNet",
#     "--model": "ResNet18",
#     "--selection": "Random",
#     # "--num_exp": 20,
#     "--num_exp": 5,
#     "--num_eval": 1,
#     # "--epochs": 2,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 1.0,
#     "--workers": 4,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
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
#     "--dataset": "THUCNews",
#     "--model": "TextCNN",
#     "--selection": "Random",
#     "--num_exp": 20,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 100,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.7,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 4096,
#     "--test_interval": 1,
#     "--selection_epochs": 12,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--balance": True
# }

# running_args = {
#     "--dataset": "SST2",
#     "--model": "TextCNN",
#     "--selection": "Random",
#     "--num_exp": 20,
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
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 2048,
#     "--test_interval": 1,
#     "--selection_epochs": 12,
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
#     "--selection": "Random",
#     "--num_exp": 20,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.7,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
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
#     "--dataset": "YELP",
#     "--model": "TextCNN",
#     "--selection": "Random",
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
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
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
#     "--dataset": "AGNews",
#     "--model": "TextCNN",
#     "--selection": "Random",
#     "--num_exp": 5,
#     # "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 100,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 1.0,
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
#     "--balance": True
# }

running_args = {
    "--dataset": "UrbanSound8K",
    "--model": "TDNN",
    "--selection": "Random",
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
    "--balance": True
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
    # wb.append('备注', 0, "Random, fraction: 1.0, model: ResNet18, dataset: CIFAR100")
    # wb.append('备注', 0, "Random, fraction: 0.01, model: LeNet, dataset: MNIST")
    # wb.append('备注', 0, "Random, fraction: 0.2, CIFAR10, ResNet18, 随机选择")
    # wb.append('备注', 0, "0.3,MNIST, LeNet")
    # wb.append('备注', 0, "fraction: 0.7, THUCNews, TextCNN")
    # wb.to_excel('./excel/data_70.xlsx')
    # wb.append('备注', 0, "fraction: 0.7, SST-5, TextCNN")
    # wb.append('备注', 0, "fraction: 0.5, YELP, TextCNN")
    # wb.append('备注', 0, "fraction: 1.0, AGNews, TextCNN")

    wb.append('备注', 0, "fraction: 0.1, UrbanSound8K, TDNN")

    # wb.append('备注', 0, "fraction: 1.0, TinyImagenet, ResNet18")
    wb.to_excel('./excel/data_10.xlsx')
    print("end")
