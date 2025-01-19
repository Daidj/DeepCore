import os

import numpy
import torch.nn as nn
import argparse

import visdom

import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime

def continual_learning(wb=None):


    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--divide_step', type=int, default=5, help='divide dataset to steps')
    parser.add_argument('--recall_epochs', type=int, default=10, help='divide dataset to steps')

    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=0, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=0.0, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help="Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                        help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                        help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy",
                        help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="LeastConfidence", help="specifiy uncertanty score to use")
    parser.add_argument('--solution_num', type=int, default=5, help="pareto solution")
    parser.add_argument('--population', type=int, default=20, help="population size")
    parser.add_argument('--step_rate', type=float, default=0.01, help="step size")
    parser.add_argument('--iter', type=int, default=30, help="solver iteration times")



    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

    args = parser.parse_args()

    vis = visdom.Visdom(env='{}_{}'.format(args.selection, args.fraction))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    checkpoint = {}
    start_exp = 0
    start_epoch = 0

    for exp in range(start_exp, args.num_exp):
        exp_start_time = time.time()
        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_exp: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device, sep="")

        final_subset = []
        for d_step in range(args.divide_step):
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](args.data_path, d_step, args.divide_step)
            args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

            torch.random.manual_seed(args.seed)

            algorithm_start_time = time.time()
            selection_args = dict(epochs=args.selection_epochs,
                                      selection_method=args.uncertainty,
                                      balance=args.balance,
                                      greedy=args.submodular_greedy,
                                      function=args.submodular
                                      )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            subsets = method.select()
            algorithm_end_time = time.time()
            global_best_prec1 = 0.0
            global_best_index = 0

            for solution in range(len(subsets)):
                subset = subsets[solution]
                selected_length = len(subset["indices"])
                print("selected length: ", selected_length)

                # Augmentation
                dst_train.transform = transforms.Compose(
                    [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                     transforms.RandomHorizontalFlip(), dst_train.transform])

                dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

                train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                           num_workers=args.workers, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                          num_workers=args.workers, pin_memory=True)

                # Listing cross-architecture experiment settings if specified.
                model = args.model

                network = nets.__dict__[model](channel, num_classes, im_size, pretrained=False).to(args.device)

                if args.device == "cpu":
                    print("Using CPU.")
                elif args.gpu is not None:
                    print("Using GPU {}".format(args.gpu[0]))
                    torch.cuda.set_device(args.gpu[0])
                    network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
                elif torch.cuda.device_count() > 1:
                    network = nets.nets_utils.MyDataParallel(network).cuda()

                criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

                # Optimizer
                if args.optimizer == "SGD":
                    optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                                weight_decay=args.weight_decay, nesterov=args.nesterov)
                elif args.optimizer == "Adam":
                    optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                     weight_decay=args.weight_decay, nesterov=args.nesterov)

                # LR scheduler
                if args.scheduler == "CosineAnnealingLR":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                           eta_min=args.min_lr)
                elif args.scheduler == "StepLR":
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                                gamma=args.gamma)
                else:
                    scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
                scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

                # Log recorder
                rec = init_recorder()

                best_prec1 = 0.0
                best_epoch = -1

                for epoch in range(start_epoch, args.epochs):
                    # train for one epoch
                    start_time = time.time()
                    train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
                    end_time = time.time()
                    print("Train time: {}".format(end_time-start_time))

                    # evaluate on validation set
                    if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                        prec1 = test(test_loader, network, criterion, epoch, args, rec)

                        # remember best prec@1 and save checkpoint
                        is_best = prec1 > best_prec1

                        if is_best:
                            best_prec1 = prec1
                            best_epoch = epoch

                print('| Best accuracy: ', best_prec1, "\nBest epoch: ", best_epoch, " on model " + model, end="\n\n")

                start_epoch = 0
                checkpoint = {}
                # visdom
                epoch_list =[i for i in range(start_epoch, args.epochs)]
                vis.line(Y=rec.train_loss, X=epoch_list, win='train_loss_{}_{}'.format(exp, solution), opts = dict(title='train_loss_{}_{}'.format(exp, solution), showlegend=True))
                vis.line(Y=rec.train_acc, X=epoch_list, win='train_acc_{}_{}'.format(exp, solution), opts=dict(title='train_acc_{}_{}'.format(exp, solution), showlegend=True))
                vis.line(Y=rec.lr, X=epoch_list, win='train_lr_{}_{}'.format(exp, solution),
                         opts=dict(title='train_lr_{}_{}'.format(exp, solution), showlegend=True))
                vis.line(Y=rec.test_loss, X=epoch_list, win='test_loss_{}_{}'.format(exp, solution),
                         opts=dict(title='test_loss_{}_{}'.format(exp, solution), showlegend=True))
                vis.line(Y=rec.test_acc, X=epoch_list, win='test_acc_{}_{}'.format(exp, solution),
                         opts=dict(title='test_acc_{}_{}'.format(exp, solution), showlegend=True))
                vis.text("Exp {} Solution {} result: Best accuracy: {}, Best epoch: {} \n".format(exp, solution, best_prec1, best_epoch), win='result', append=False if exp + solution == 0 else True)
                if wb != None:
                    wb.append('solution_{}'.format(solution), exp, best_prec1)
                if best_prec1 > global_best_prec1:
                    global_best_prec1 = best_prec1
                    global_best_index = solution
                    # test_data_folder = 'test_data/iter_{}_{}/multi_{}'.format(args.selection, args.iter, args.dataset)
                    # os.makedirs(test_data_folder, exist_ok=True)
                    # best_file_path = os.path.join(test_data_folder, 'best_{}.npy'.format(args.fraction))
                    # best_index_path = os.path.join(test_data_folder, 'best_index_{}.npy'.format(args.fraction))
                    # numpy.save(best_file_path, subset["indices"])
                    # numpy.save(best_index_path, numpy.array([solution]))
            exp_end_time = time.time()
            vis.text("Exp {} result: Best accuracy: {}, Best index: {} \n".format(exp,global_best_prec1, global_best_index), win='result', append=True)

            if wb != None:
                wb.append('样本数量', exp, selected_length)
                wb.append('总时间', exp, exp_end_time - exp_start_time)
                wb.append('准确度', exp, global_best_prec1)
                wb.append('算法时间', exp, algorithm_end_time - algorithm_start_time)
            final_subset.append(subsets[global_best_index]["indices"])

        model = args.model
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](
            args.data_path, 0, args.divide_step)

        network = nets.__dict__[model](channel, num_classes, im_size, pretrained=False).to(args.device)
        global_best_prec1 = 0.0
        if args.device == "cpu":
            print("Using CPU.")
        elif args.gpu is not None:
            print("Using GPU {}".format(args.gpu[0]))
            torch.cuda.set_device(args.gpu[0])
            network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
        elif torch.cuda.device_count() > 1:
            network = nets.nets_utils.MyDataParallel(network).cuda()

        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

        for d_step in range(args.divide_step):
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[
                args.dataset](args.data_path, d_step, args.divide_step)
            args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

            torch.random.manual_seed(args.seed)

            # Augmentation
            dst_train.transform = transforms.Compose(
                [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                 transforms.RandomHorizontalFlip(), dst_train.transform])

            # dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])

            train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=True)

            # Listing cross-architecture experiment settings if specified.


            # Optimizer
            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            # LR scheduler
            if args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                       eta_min=args.min_lr)
            elif args.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                            gamma=args.gamma)
            else:
                scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
            scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

            # Log recorder
            if "rec" in checkpoint.keys():
                rec = checkpoint["rec"]
            else:
                rec = init_recorder()

            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0
            best_epoch = -1

            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                start_time = time.time()
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
                end_time = time.time()
                print("Train time: {}".format(end_time - start_time))

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        best_epoch = epoch

            print('| Best accuracy: ', best_prec1, "\nBest epoch: ", best_epoch, " on model " + model, end="\n\n")

            start_epoch = 0
            checkpoint = {}
            # visdom
            epoch_list = [i for i in range(start_epoch, args.epochs)]
            vis.line(Y=rec.train_loss, X=epoch_list, win='train_loss_{}_{}'.format(exp, d_step),
                     opts=dict(title='train_loss_{}_{}'.format(exp, d_step), showlegend=True))
            vis.line(Y=rec.train_acc, X=epoch_list, win='train_acc_{}_{}'.format(exp, d_step),
                     opts=dict(title='train_acc_{}_{}'.format(exp, d_step), showlegend=True))
            vis.line(Y=rec.lr, X=epoch_list, win='train_lr_{}_{}'.format(exp, d_step),
                     opts=dict(title='train_lr_{}_{}'.format(exp, d_step), showlegend=True))
            vis.line(Y=rec.test_loss, X=epoch_list, win='test_loss_{}_{}'.format(exp, d_step),
                     opts=dict(title='test_loss_{}_{}'.format(exp, d_step), showlegend=True))
            vis.line(Y=rec.test_acc, X=epoch_list, win='test_acc_{}_{}'.format(exp, d_step),
                     opts=dict(title='test_acc_{}_{}'.format(exp, d_step), showlegend=True))
            vis.text("Exp {} step {} result: Best accuracy: {}, Best epoch: {} \n".format(exp, d_step, best_prec1,
                                                                                              best_epoch), win='result',
                     append=True)
            if best_prec1 > global_best_prec1:
                global_best_prec1 = best_prec1
                # global_best_index = solution
                # test_data_folder = 'test_data/iter_{}_{}/multi_{}'.format(args.selection, args.iter, args.dataset)
                # os.makedirs(test_data_folder, exist_ok=True)
                # best_file_path = os.path.join(test_data_folder, 'best_{}.npy'.format(args.fraction))
                # best_index_path = os.path.join(test_data_folder, 'best_index_{}.npy'.format(args.fraction))
                # numpy.save(best_file_path, subset["indices"])
                # numpy.save(best_index_path, numpy.array([solution]))
        exp_end_time = time.time()
        vis.text(
            "Exp {} result: Best accuracy: {} \n".format(exp, global_best_prec1),
            win='result', append=True)

        # 回忆
        merge_dataset = None
        for d_step in range(args.divide_step):
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[
                args.dataset](args.data_path, d_step, args.divide_step)
            if merge_dataset is None:
                channel, im_size, num_classes, class_names, mean, std, merge_dataset, dst_test = datasets.__dict__[
                    'Merge'](args.data_path, dst_train, final_subset[d_step])
            else:
                merge_dataset.merge(dst_train, final_subset[d_step])
        dst_subset = merge_dataset
        print('meger dataset: ', len(dst_train))
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        torch.random.manual_seed(args.seed)

        # Augmentation
        dst_train.transform = transforms.Compose(
            [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
             transforms.RandomHorizontalFlip(), dst_train.transform])

        # dst_subset = torch.utils.data.Subset(dst_train, final_subset[d_step])

        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

            # Listing cross-architecture experiment settings if specified.

            # Optimizer
            # if args.optimizer == "SGD":
            #     optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
            #                                 weight_decay=args.weight_decay, nesterov=args.nesterov)
            # elif args.optimizer == "Adam":
            #     optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            # else:
            #     optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
            #                                                      weight_decay=args.weight_decay, nesterov=args.nesterov)

            # LR scheduler
            # if args.scheduler == "CosineAnnealingLR":
            #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
            #                                                            eta_min=args.min_lr)
            # elif args.scheduler == "StepLR":
            #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
            #                                                 gamma=args.gamma)
            # else:
            #     scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
            # scheduler.last_epoch = (start_epoch - 1) * len(train_loader)

            # Log recorder
        if "rec" in checkpoint.keys():
            rec = checkpoint["rec"]
        else:
            rec = init_recorder()

        best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0
        best_epoch = -1

        for epoch in range(start_epoch, args.recall_epochs):
            # train for one epoch
            start_time = time.time()
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            end_time = time.time()
            print("Train time: {}".format(end_time - start_time))

            # evaluate on validation set
            if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                prec1 = test(test_loader, network, criterion, epoch, args, rec)

                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1

                if is_best:
                    best_prec1 = prec1
                    best_epoch = epoch

        print('| Best accuracy: ', best_prec1, "\nBest epoch: ", best_epoch, " on model " + model, end="\n\n")

        start_epoch = 0
        checkpoint = {}
        # visdom
        epoch_list = [i for i in range(start_epoch, args.recall_epochs)]
        vis.line(Y=rec.train_loss, X=epoch_list, win='train_loss_{}_{}'.format(exp, solution),
                 opts=dict(title='train_loss_{}_{}'.format(exp, solution), showlegend=True))
        vis.line(Y=rec.train_acc, X=epoch_list, win='train_acc_{}_{}'.format(exp, solution),
                 opts=dict(title='train_acc_{}_{}'.format(exp, solution), showlegend=True))
        vis.line(Y=rec.lr, X=epoch_list, win='train_lr_{}_{}'.format(exp, solution),
                 opts=dict(title='train_lr_{}_{}'.format(exp, solution), showlegend=True))
        vis.line(Y=rec.test_loss, X=epoch_list, win='test_loss_{}_{}'.format(exp, solution),
                 opts=dict(title='test_loss_{}_{}'.format(exp, solution), showlegend=True))
        vis.line(Y=rec.test_acc, X=epoch_list, win='test_acc_{}_{}'.format(exp, solution),
                 opts=dict(title='test_acc_{}_{}'.format(exp, solution), showlegend=True))
        vis.text("Exp {} Solution {} result: Best accuracy: {}, Best epoch: {} \n".format(exp, solution, best_prec1,
                                                                                          best_epoch), win='result',
                 append=True)
        if wb != None:
            wb.append('solution_{}'.format(solution), exp, best_prec1)
        if best_prec1 > global_best_prec1:
            global_best_prec1 = best_prec1
            global_best_index = solution
            # test_data_folder = 'test_data/iter_{}_{}/multi_{}'.format(args.selection, args.iter, args.dataset)
            # os.makedirs(test_data_folder, exist_ok=True)
            # best_file_path = os.path.join(test_data_folder, 'best_{}.npy'.format(args.fraction))
            # best_index_path = os.path.join(test_data_folder, 'best_index_{}.npy'.format(args.fraction))
            # numpy.save(best_file_path, subset["indices"])
            # numpy.save(best_index_path, numpy.array([solution]))
        exp_end_time = time.time()
        vis.text(
            "Exp {} result: Best accuracy: {} \n".format(exp, global_best_prec1),
            win='result', append=True)

        if wb != None:
            # wb.append('样本数量', exp, selected_length)
            wb.append('总时间', exp, exp_end_time - exp_start_time)
            wb.append('准确度', exp, global_best_prec1)
            # wb.append('算法时间', exp, algorithm_end_time - algorithm_start_time)



if __name__ == '__main__':
    global_start_time = time.time()
    # multi_main()
    global_end_time = time.time()
    print("Time: ", global_end_time-global_start_time)