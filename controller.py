import json
import os
import sys

from WorkBook import WorkBook
from multi_main import multi_main


class Controller:
    _instance = None  # 类变量，用于存储单例实例
    config_path = 'config'

    logger = []

    def __init__(self):
        self.output = 'output'
        with open(os.path.join(self.config_path, "dataset.json"), "r") as file:
            self.dataset_info = json.load(file)
        with open(os.path.join(self.config_path, "algorithm.json"), "r") as file:
            self.alg_info = json.load(file)

    @classmethod
    def getController(cls):
        if cls._instance is None:
            cls._instance = Controller()
        return cls._instance

    def run_algorithm(self, args):
        running_args = {
            "--dataset": "CIFAR10",
            "--model": "ResNet18",
            "--selection": "MOEA2",
            "--num_exp": 1,
            "--num_eval": 1,
            # "--epochs": 200,
            "--epochs": 2,
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
            "--selection_epochs": 1,
            "--selection_momentum": 0.0,
            "--selection_weight_decay": 0.0,
            "--selection_optimizer": "Adam",
            "--selection_lr": 0.002,
            "--selection_test_interval": 1,
            "--uncertainty": "Entropy",
            "--balance": True,
            "--solution_num": 5,
            "--population": 10,
            "--step_rate": 0.1,
            "--iter": 20
        }

        for key, value in args.items():
            running_args[key] = value

        wb = WorkBook(running_args["--num_exp"])
        print(sys.argv)
        for key, value in running_args.items():
            sys.argv.append(key)
            sys.argv.append(str(value))
        print(sys.argv)
        self.log(str(args))
        result = multi_main(wb)
        wb.append('备注', 0,
                  "MOEA2, fraction: 0.1, model: ResNet18, dataset: CIFAR10, MOEA2")

        wb.to_excel(os.path.join(self.output, 'data.xlsx'))
        self.log('Evaluate CoreSet Accuracy: {}'.format(result))


    def add_dataset(self, args):
        info = self.dataset_info[args['dataset type']]['dataset']
        if args['dataset'] in info.keys():
            return False
        info[args['dataset']] = {
            'data path': args['dataset path'],
            'number of classes': args['number of classes']
        }
        with open(os.path.join(self.config_path, "dataset.json"), "w") as file:
            json.dump(self.dataset_info, file)
        return True

    def log(self, text):
        for l in self.logger:
            l(text)

    def set_output_directory(self, path):
        os.makedirs(path, exist_ok=True)
        self.output = path

    def get_dataset_type_list(self):
        return self.dataset_info.keys()

    def get_dataset_list(self, type):
        return self.dataset_info[type]['dataset'].keys()

    def get_model_list(self, type):
        return self.dataset_info[type]['model']

    def get_algorithm_list(self):
        return self.alg_info.keys()


# 测试单例模式
if __name__ == "__main__":
    config_path = 'config'
    # info = {
    #     'Image': {
    #         'dataset': {
    #             'MNIST': {
    #                 'data path': 'data',
    #                 'number of classes': 10
    #             },
    #             'CIFAR10': {
    #                 'data path': 'data',
    #                 'number of classes': 10
    #             },
    #             'CIFAR100': {
    #                 'data path': 'data',
    #                 'number of classes': 100
    #             },
    #         },
    #         'model': ['LeNet', 'ResNet18'],
    #     },
    #     'Text': {
    #         'dataset': {
    #             'SST5': {
    #                 'data path': 'data',
    #                 'number of classes': 5
    #             },
    #         },
    #         'model': ['TextCNN'],
    #     },
    #     'Audio': {
    #         'dataset': {
    #             'UrbanSound8K': {
    #                 'data path': 'data',
    #                 'number of classes': 10
    #             },
    #         },
    #         'model': ['TDNN'],
    #     },
    # }
    # os.makedirs(config_path, exist_ok=True)
    # with open(os.path.join(config_path, "data.json"), "w") as file:
    #     json.dump(info, file)  #

    # with open(os.path.join(config_path, "dataset.json"), "r") as file:
    #     loaded_data = json.load(file)
    # print(loaded_data)

    alg_info = {
        'MicroSearch': {

        },
        'kCenterGreedy': {

        }
    }
    os.makedirs(config_path, exist_ok=True)
    with open(os.path.join(config_path, "algorithm.json"), "w") as file:
        json.dump(alg_info, file)  #
