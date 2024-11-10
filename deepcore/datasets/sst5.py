import os
from typing import Any

import gensim
import pyarrow.parquet as pq
import nltk
import pickle
import torch
import numpy as np
from gensim import downloader
from torch.utils.data import Dataset
from gensim.models import word2vec

class SST5DataSet(Dataset):
    def __init__(self, root: str, word_2_index: dict, train: bool = True, train_valid_merge=True):
        self.root_dir = root
        self.train = train
        self.train_valid_merge = train_valid_merge
        self.data: Any = []
        self.targets = []
        self.classes = [x for x in range(5)]
        self.dev_data: Any = []
        self.dev_targets = []
        self.word_2_index = word_2_index
        self.max_len = 53
        self.init_dataset()
        self.num_classes = len(self.classes)  # 类别数

        self.n_vocab = len(word_2_index)

        print("SST5 Dataset init")

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
        parquet_file = pq.ParquetFile(data_path)
        table = parquet_file.read()
        sentences = table.column('text')
        labels = table.column('label')
        all_labels = np.concatenate(labels.chunks)
        sentences_list = []
        for chunk in sentences.chunks:
            sentences_list.extend(chunk.to_pandas().tolist())

        all_texts = []
        for data in sentences_list:
            words = str(data).split(' ')
            words = words[:self.max_len]
            text_idx = [self.word_2_index.get(i, 0) for i in words]
            text_idx = text_idx + [1] * (self.max_len - len(words))
            all_texts.append(text_idx)

        return np.array(all_texts), np.array(all_labels)

    def init_dataset(self):
        folder_name = self.root_dir

        if self.train:

            data_path = os.path.join(folder_name, 'train.parquet')
            self.data, self.targets = self.read_data(data_path)
            dev_data_path = os.path.join(folder_name, 'validation.parquet')
            self.dev_data, self.dev_targets = self.read_data(dev_data_path)
            if self.train_valid_merge:
                self.data = np.vstack((self.data, self.dev_data))  # 垂直堆叠
                self.targets = np.hstack((self.targets, self.dev_targets))
        else:
            data_path = os.path.join(folder_name, 'test.parquet')
            self.data, self.targets = self.read_data(data_path)

def build_vocab(file_path):
    word_2_index = {'UNK': 0, 'PAD': 1}
    parquet_file = pq.ParquetFile(file_path)
    table = parquet_file.read()
    sentences = table.column('text')
    sentences_list = []
    for chunk in sentences.chunks:
        sentences_list.extend(chunk.to_pandas().tolist())
    max_length = 0
    for data in sentences_list:
        words = nltk.word_tokenize(str(data))
        if (len(words) > max_length):
            max_length = len(words)
        for word in words:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    print(max_length)
    return word_2_index

def build_vocab_genism(file_path):
    # size: 300
    model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
    print(model['man'])
    word_2_index = {'UNK': 0, 'PAD': 1}
    parquet_file = pq.ParquetFile(file_path)
    table = parquet_file.read()
    sentences = table.column('text')
    sentences_list = []
    for chunk in sentences.chunks:
        sentences_list.extend(chunk.to_pandas().tolist())
    max_length = 0
    for data in sentences_list:
        words = nltk.word_tokenize(str(data))
        if (len(words) > max_length):
            max_length = len(words)
        for word in words:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    print(max_length)
    return word_2_index

def SST5(data_path):
    channel = None
    im_size = None
    mean = None
    std = None
    folder_name = os.path.join(data_path, 'SST-5')
    vocab_path = os.path.join(folder_name, 'character_vocab_nltk.pkl')
    if os.path.exists(vocab_path):
        # 文件存在,则加载 .pkl 文件
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocab file loaded successfully!")
    else:
        vocab = build_vocab(os.path.join(folder_name, 'train.parquet'))
        pickle.dump(vocab, open(vocab_path, 'wb'))

    dst_train = SST5DataSet(root=folder_name, word_2_index=vocab, train=True)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_test = SST5DataSet(root=folder_name, word_2_index=vocab, train=False)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def SST5_new(data_path):
    channel = None
    im_size = None
    mean = None
    std = None
    folder_name = os.path.join(data_path, 'SST-5')
    vocab_path = os.path.join(data_path, 'GoogleNews-vectors-negative300.bin')
    vocab = build_vocab_genism(vocab_path)

    dst_train = SST5DataSet(root=folder_name, word_2_index=vocab, train=True)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_test = SST5DataSet(root=folder_name, word_2_index=vocab, train=False)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

if __name__ == '__main__':
    SST5('/home/sample_selection/data')