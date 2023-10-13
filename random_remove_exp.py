import sys

from WorkBook import WorkBook
from main import main

running_args = {
    "--dataset": "MNIST",
    "--model": "LeNet",
    "--selection": "Random",
    "--num_exp": 20,
    "--num_eval": 1,
    "--epochs": 200,
    "--data_path": "data",
    "--gpu": 0,
    "--print_freq": 20,
    "--fraction": 1.0,
    "--workers": 8,
    "--optimizer": "Adam",
    "--lr": 0.002,
    "--momentum": 0.0,
    "--weight_decay": 0.0,
    "--nesterov": False,
    "--train_batch": 256,
    "--test_interval": 1
}

if __name__ == '__main__':
    start_fraction = 0.1375
    number = 1
    fraction_list = [start_fraction - i * 0.02 for i in range(number)]
    # result_dict = {'实验序号/选择比例': [i+1 for i in range(running_args["--num_eval"])]}
    wb = WorkBook(running_args["--num_exp"], fraction_list)

    origin_argv = sys.argv
    print(sys.argv)
    for fraction in fraction_list:
        running_args["--fraction"] = fraction
        for key, value in running_args.items():
            sys.argv.append(key)
            sys.argv.append(str(value))
        print(sys.argv)
        main(wb)

        sys.argv = []
        sys.argv.append(origin_argv[0])
    wb.to_excel('./excel/data_{}.xlsx'.format(start_fraction))
    print("random end")
