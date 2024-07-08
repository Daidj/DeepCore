import os
from typing import Any

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


# https://github.com/649453932/Chinese-Text-Classification-Pytorch


class THUCNewsDataset(Dataset):
    def __init__(self, root: str, word_2_index: dict, train: bool = True):
        self.root_dir = root
        self.train = train
        self.data: Any = []
        self.targets = []
        self.classes = []
        self.dev_data: Any = []
        self.dev_targets = []
        self.word_2_index = word_2_index
        self.max_len = 32
        self.init_dataset()
        self.num_classes = len(self.classes)  # 类别数

        self.n_vocab = len(word_2_index)

        print("THUCNews Dataset init")

    def __getitem__(self, index):
        text, target = self.data[index], self.targets[index]
        # text = self.data[item][:self.max_len]
        # text_idx = [self.word_2_index.get(i, 0) for i in text]
        # text_idx = text_idx + [1] * (self.max_len - len(text))
        # label = int(self.targets[item])
        return torch.tensor(text), torch.tensor(target)

    def __len__(self):
        return len(self.data)

    def read_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = f.read().split('\n')
        all_texts = []
        all_labels = []
        for data in all_data:
            if data:
                t, l = data.split('\t')
                text = t[:self.max_len]
                text_idx = [self.word_2_index.get(i, 0) for i in text]
                text_idx = text_idx + [1] * (self.max_len - len(text))
                label = int(l)
                all_texts.append(text_idx)
                all_labels.append(label)

        return np.array(all_texts), np.array(all_labels)

    def init_dataset(self):
        folder_name = self.root_dir
        self.classes = [x.strip() for x in
                        open(os.path.join(folder_name, 'class.txt'), encoding='utf-8').readlines()]
        if self.train:
            data_path = os.path.join(folder_name, 'train.txt')
            self.data, self.targets = self.read_data(data_path)
            dev_data_path = os.path.join(folder_name, 'dev.txt')
            self.dev_data, self.dev_targets = self.read_data(dev_data_path)
        else:
            data_path = os.path.join(folder_name, 'test.txt')
            self.data, self.targets = self.read_data(data_path)

def build_vocab(file_path):
    word_2_index = {'UNK': 0, 'PAD': 1}
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = f.read().split('\n')
    for data in all_data:
        if data:
            text, label = data.split('\t')
            for word in text:
                if word not in word_2_index:
                    word_2_index[word] = len(word_2_index)
    return word_2_index


def THUCNews(data_path):
    channel = None
    im_size = None
    mean = None
    std = None
    folder_name = os.path.join(data_path, 'THUCNEWS')
    vocab_path = os.path.join(folder_name, 'character_vocab.pkl')
    if os.path.exists(vocab_path):
        # 文件存在,则加载 .pkl 文件
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocab file loaded successfully!")
    else:
        vocab = build_vocab(os.path.join(folder_name, 'train.txt'))
        pickle.dump(vocab, open(vocab_path, 'wb'))

    dst_train = THUCNewsDataset(root=folder_name, word_2_index=vocab, train=True)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_test = THUCNewsDataset(root=folder_name, word_2_index=vocab, train=False)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

