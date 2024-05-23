import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model

# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "Random",
#     # "--num_exp": 20,
#     "--num_exp": 1,
#     "--num_eval": 1,
#     "--epochs": 2,
#     # "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 1.0,
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

running_args = {
    "--dataset": "CIFAR10",
    "--model": "ResNet18",
    "--selection": "Random",
    # "--num_exp": 20,
    "--num_exp": 1,
    "--num_eval": 1,
    "--epochs": 2,
    # "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 1.0,
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
    wb.append('备注', 0, "全部数据,CIFAR10, ResNet18")
    # wb.append('备注', 0, "全部数据,MNIST, ResNet")
    wb.to_excel('./excel/data_full.xlsx')
    print("end")
