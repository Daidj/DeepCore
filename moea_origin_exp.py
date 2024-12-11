import sys

from WorkBook import WorkBook
from multi_main import multi_main
from random_remove_exp import test_model

running_args = {
    "--dataset": "MNIST",
    "--model": "LeNet",
    "--selection": "MOEAD",
    # "--num_exp": 2,
    "--num_exp": 1,
    "--num_eval": 1,
    # "--epochs": 20,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.5,
    "--workers": 8,
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
    "--uncertainty": "Entropy",
    "--balance": True,
    # "--solution_num": 2,
    "--population": 20,
    # "--step_rate": 0.3,
    # "--iter": 1,
    "--solution_num": 5,
    "--step_rate": 0.1,
    "--iter": 50
}

# running_args = {
#     "--dataset": "CIFAR10",
#     "--model": "ResNet18",
#     "--selection": "MOEAD",
#     # "--num_exp": 1,
#     "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.1,
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
#     "--uncertainty": "Entropy",
#     "--balance": True,
#     "--solution_num": 5,
#     "--step_rate": 0.1,
#     "--iter": 50
# }

# running_args = {
#     "--dataset": "CIFAR100",
#     "--model": "ResNet18",
#     "--selection": "MOEAD",
#     # "--num_exp": 20,
#     "--num_exp": 1,
#     "--num_eval": 1,
#     # "--epochs": 2,
#     "--epochs": 200,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.7,
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
#     "--balance": True,
#     "--population": 20,
#     "--solution_num": 5,
#     "--step_rate": 0.1,
#     "--iter": 50
# }

# running_args = {
#     "--dataset": "SST5",
#     "--model": "TextCNN",
#     "--selection": "MOEAD",
#     # "--num_exp": 20,
#     "--num_exp": 1,
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
#     "--solution_num": 5
# }

# running_args = {
#     "--dataset": "YELP",
#     "--model": "TextCNN",
#     "--selection": "MOEAD",
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
#     "--solution_num": 5

# }

# running_args = {
#     "--dataset": "AGNews",
#     "--model": "TextCNN",
#     "--selection": "MOEAD",
#     # "--num_exp": 5,
#     "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 10,
#     "--epochs": 100,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.1,
#     "--workers": 8,
#     "--optimizer": "Adam",
#     "--lr": 0.001,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 256,
#     "--selection_batch": 256,
#     "--test_interval": 1,
#     # "--selection_epochs": 1,
#     "--selection_epochs": 12,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.001,
#     "--selection_test_interval": 1,
#     "--balance": True,
#     "--solution_num": 5
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
    multi_main(wb)
    wb.append('备注', 0, "MOEAD, fraction: 0.5, model: LeNet, dataset: MNIST, last_layer, Info, ratio: 1.0, cosine, iter: 50, population: 20, step_rate: 0.1, mmd: 0.003")
    wb.to_excel('./excel/data_MOEAD_50.xlsx')
    # wb.append('备注', 0, "MOEAD, fraction: 0.1, model: ResNet18, dataset: CIFAR10, MOEAD, last_layer, Info, ratio: 1.0, cosine, iter: 50, step_rate: 0.1, mmd: 0.003")
    # wb.to_excel('./excel/data_MOEAD_10_2.xlsx')
    # wb.append('备注', 0, "MOEAD, fraction: 0.7, model: ResNet18, dataset: CIFAR100, MOEAD, last_layer, Info, ratio: 1.0, cosine, iter: 50, population: 20, step_rate: 0.1, mmd: 0.003")
    # wb.to_excel('./excel/data_MOEAD_70_2.xlsx')
    # wb.append('备注', 0, "MOEAD, fraction: 0.1, model: TextCNN, dataset: SST-5, uniqueness+kcenter(一致的归一化), batch: 256, 比例优化空间, "
    #                    "0.5:0.5, 特征矩阵:outputs, 置信度：标签索引, iter: 50")
    # wb.append('备注', 0, "MOEAD, fraction: 0.7, model: TextCNN, dataset: SST-5, ratio: 1.0, middle: 0.75")
    # wb.append('备注', 0, "MOEAD, fraction: 0.5, model: TextCNN, dataset: YELP, uniqueness+kcenter(一致的归一化), batch: 256, 比例优化空间, "
    #                    "0.5:0.5, 特征矩阵:outputs, 置信度：标签索引, iter: 50")
    # wb.append('备注', 0, "MOEAD, fraction: 0.1, model: TextCNN, dataset: AG News, MOEAD, last_layer, Confidence, ratio: 1.0, middle: 动态,min_mmd_distance: 0.002, cosine")
    # wb.to_excel('./excel/data_MOEAD_10.xlsx')
    print("end")
