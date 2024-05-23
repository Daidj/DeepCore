import sys

from WorkBook import WorkBook
from main import main
from random_remove_exp import test_model

# running_args = {
#     "--dataset": "MEDICAL",
#     "--model": "LeNet",
#     "--selection": "Random",
#     "--num_exp": 2,
#     # "--num_exp": 2,
#     "--num_eval": 1,
#     # "--epochs": 20,
#     "--epochs": 10,
#     "--data_path": "data",
#     "--gpu": 0,
#     "--print_freq": 20,
#     "--fraction": 0.2,
#     "--workers": 1,
#     "--optimizer": "Adam",
#     "--lr": 0.002,
#     "--momentum": 0.0,
#     "--weight_decay": 0.0,
#     "--nesterov": False,
#     "--train_batch": 4,
#     "--test_interval": 1,
#     "--selection_epochs": 25,
#     # "--selection_epochs": 2,
#     "--selection_momentum": 0.0,
#     "--selection_weight_decay": 0.0,
#     "--selection_optimizer": "Adam",
#     "--selection_lr": 0.002,
#     "--selection_test_interval": 1,
#     "--uncertainty": "Confidence",
#     "--balance": True
# }

running_args = {
    "--dataset": "MEDICAL",
    "--model": "LeNet",
    "--selection": "UncertaintyKcenterDenoise",
    "--num_exp": 2,
    # "--num_exp": 2,
    "--num_eval": 1,
    # "--epochs": 20,
    "--epochs": 10,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 0.8,
    "--workers": 0,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 4,
    "--test_interval": 1,
    "--selection_epochs": 2,
    # "--selection_epochs": 2,
    "--batch": 4,
    "--selection_momentum": 0.0,
    "--selection_weight_decay": 0.0,
    "--selection_optimizer": "Adam",
    "--selection_lr": 0.002,
    "--selection_test_interval": 1,
    "--uncertainty": "LeastConfidence",
    "--balance": True
}

# running_args = {
#     "--dataset": "CIFAR10",
#     "--model": "ResNet18",
#     "--selection": "Uncertainty",
#     "--num_exp": 5,
#     # "--num_exp": 2,
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
    # wb.append('备注', 0, "Random, fraction: 0.2, model: LeNet, dataset:MEDICAL")
    # wb.to_excel('./excel/data_random_medical_02.xlsx')
    wb.append('备注', 0, "uncertaintyKcenterDenoise, LeastConfidence, fraction: 0.8, model: LeNet, dataset:MEDICAL, 去除噪声元素后的样本选择,使用软修剪，噪声削弱比例2,多样性扩充比例: 1.25")

    wb.to_excel('./excel/data_uncertaintyKcenterDenoise_LeastConfidence_08.xlsx')
    print("end")
