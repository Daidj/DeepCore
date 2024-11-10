import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model
#
# running_args = {
#     "--dataset": "MNIST",
#     "--model": "LeNet",
#     "--selection": "Forgetting",
#     # "--num_exp": 20,
#     "--num_exp": 5,
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
#
# running_args = {
#     "--dataset": "CIFAR10",
#     "--model": "ResNet18",
#     "--selection": "Forgetting",
#     # "--num_exp": 3,
#     "--num_exp": 5,
#     "--num_eval": 1,
#     # "--epochs": 20,
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
#     # "--selection_epochs": 1,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "LeastConfidence",
#     "--balance": True
# }

running_args = {
    "--dataset": "CIFAR100",
    "--model": "ResNet18",
    "--selection": "Forgetting",
    # "--num_exp": 20,
    "--num_exp": 5,
    "--num_eval": 1,
    # "--epochs": 2,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.9,
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
    # wb.append('备注', 0, "Forgetting, fraction: 0.1, model: LeNet, dataset: MNIST")
    # wb.to_excel('./excel/data_Forgetting_10_50_50.xlsx')
    # wb.append('备注', 0, "Forgetting, fraction: 0.7, model: ResNet18, dataset: CIFAR10, uniqueness+kcenter(一致的归一化), batch: 256, 比例优化空间, "
    #                    "0.5:0.5, 特征矩阵:outputs, 置信度：标签索引")
    # wb.to_excel('./excel/data_Forgetting_70_50_50.xlsx')
    wb.append('备注', 0, "Forgetting, fraction: 0.9, model: ResNet18, dataset: CIFAR100")
    wb.to_excel('./excel/data_forgetting_90.xlsx')
    print("end")
    # 三点确定一个圆？