import os
from typing import Any

import sklearn
from sklearn.datasets import fetch_20newsgroups
import gensim
import pyarrow.parquet as pq
import nltk
import pickle
import torch
import numpy as np
from gensim import downloader
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.utils.data import Dataset
from gensim.models import word2vec

class NewsGroupsDataSet(Dataset):
    def __init__(self, root: str, word_2_index: dict, train: bool = True, train_valid_merge=True):
        self.root_dir = root
        self.train = train
        self.data: Any = []
        self.targets = []
        self.classes = [x for x in range(20)]
        self.dev_data: Any = []
        self.dev_targets = []
        self.word_2_index = word_2_index
        self.max_len = 78881
        self.init_dataset()
        self.num_classes = len(self.classes)  # 类别数

        self.n_vocab = len(word_2_index)

        print("NewsGroups Dataset init")

    def __getitem__(self, index):
        text, target = self.data[index], self.targets[index]
        # text = self.data[item][:self.max_len]
        # text_idx = [self.word_2_index.get(i, 0) for i in text]
        # text_idx = text_idx + [1] * (self.max_len - len(text))
        # label = int(self.targets[item])
        return torch.tensor(text), torch.tensor(target)

    def __len__(self):
        return len(self.data)

    # def init_dataset(self):
    #     if self.train:
    #         newsgroups_train = fetch_20newsgroups(data_home=self.root_dir, subset='train', remove=('headers', 'footers', 'quotes'))
    #         texts = newsgroups_train.data
    #         self.targets = np.array(newsgroups_train.target)
    #         # 使用 TF-IDF 向量化
    #         vectorizer = TfidfVectorizer(max_features=1000)  # 限制特征数量
    #         tfidf_matrix = vectorizer.fit_transform(texts)
    #
    #         # 转换为密集矩阵
    #         dense_matrix = tfidf_matrix.toarray()
    #
    #         self.data = np.array(dense_matrix)
    #
    #     else:
    #         newsgroups_train = fetch_20newsgroups(data_home=self.root_dir, subset='test',
    #                                               remove=('headers', 'footers', 'quotes'))
    #         texts = newsgroups_train.data
    #         self.targets = np.array(newsgroups_train.target)
    #         # 使用 TF-IDF 向量化
    #         vectorizer = TfidfVectorizer(max_features=1000)  # 限制特征数量
    #         tfidf_matrix = vectorizer.fit_transform(texts)
    #
    #         # 转换为密集矩阵
    #         dense_matrix = tfidf_matrix.toarray()
    #
    #         self.data = np.array(dense_matrix)

    def init_dataset(self):
        if self.train:
            if not os.path.exists(os.path.join(self.root_dir, 'train_samples.npy')):
                newsgroups_train = fetch_20newsgroups(data_home=self.root_dir, subset='train', remove=('headers', 'footers', 'quotes'))
                texts = newsgroups_train.data
                self.targets = np.array(newsgroups_train.target)
                all_texts = []
                for data in texts:
                    words = nltk.word_tokenize(str(data))
                    words = words[:self.max_len]
                    text_idx = [self.word_2_index.get(i, 0) for i in words]
                    text_idx = text_idx + [1] * (self.max_len - len(words))
                    all_texts.append(text_idx)
                self.data = np.array(all_texts)

                np.save(os.path.join(self.root_dir, 'train_samples.npy'), self.data)
                np.save(os.path.join(self.root_dir, 'train_labels.npy'), self.targets)
            else:
                self.data = np.load(os.path.join(self.root_dir, 'train_samples.npy'))
                self.targets = np.load(os.path.join(self.root_dir, 'train_labels.npy'))

        else:
            if not os.path.exists(os.path.join(self.root_dir, 'test_samples.npy')):

                newsgroups_train = fetch_20newsgroups(data_home=self.root_dir, subset='test',
                                                      remove=('headers', 'footers', 'quotes'))
                texts = newsgroups_train.data
                self.targets = np.array(newsgroups_train.target)
                all_texts = []
                for data in texts:
                    words = nltk.word_tokenize(str(data))
                    words = words[:self.max_len]
                    text_idx = [self.word_2_index.get(i, 0) for i in words]
                    text_idx = text_idx + [1] * (self.max_len - len(words))
                    all_texts.append(text_idx)
                self.data = np.array(all_texts)

                np.save(os.path.join(self.root_dir, 'test_samples.npy'), self.data)
                np.save(os.path.join(self.root_dir, 'test_labels.npy'), self.targets)
            else:
                self.data = np.load(os.path.join(self.root_dir, 'test_samples.npy'))
                self.targets = np.load(os.path.join(self.root_dir, 'test_labels.npy'))


def build_vocab(file_path):
    word_2_index = {'UNK': 0, 'PAD': 1}
    newsgroups = fetch_20newsgroups(data_home=file_path, subset='all', remove=('headers', 'footers', 'quotes'))
    sentences_list = newsgroups.data
    max_length = 0
    for data in sentences_list:
        words = nltk.word_tokenize(str(data))
        if (len(words) > max_length):
            max_length = len(words)
        for word in words:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    print(max_length)
    print(len(word_2_index))
    return word_2_index

def NewsGroups(data_path):
    channel = None
    im_size = None
    mean = None
    std = None
    folder_name = os.path.join(data_path, '20NewsGroups')
    vocab_path = os.path.join(folder_name, 'character_vocab_nltk.pkl')
    if os.path.exists(vocab_path):
        # 文件存在,则加载 .pkl 文件
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocab file loaded successfully!")
    else:
        vocab = build_vocab(folder_name)
        pickle.dump(vocab, open(vocab_path, 'wb'))
    dst_train = NewsGroupsDataSet(root=folder_name, word_2_index=vocab, train=True)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_test = NewsGroupsDataSet(root=folder_name, word_2_index=vocab, train=False)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

if __name__ == '__main__':
    # nltk.download()
    NewsGroups('/home/tmp_sample/data')

    # 加载全部数据
    # 最简单的开始
    # import gensim
    # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    #
    # sentences = newsgroups_train.data
    #
    # vocab_path = os.path.join('/home/tmp_sample/data', 'GoogleNews-vectors-negative300.bin')
    # # 模型训练
    # model = gensim.models.KeyedVectors.load_word2vec_format(va, binary=True)
    #
    # print(model.similarity(sentences[0]))
    # min_count,频数阈值，大于等于1的保留
    # size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
    # workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。

    # 加载训练数据
    # newgroups = sklearn.datasets.fetch_20newsgroups_vectorized(subset='all', remove=('headers', 'footers', 'quotes')
    # newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    # labels = newsgroups_train.target
    # texts = newsgroups_train.data
    #
    #
    # # 使用 TF-IDF 向量化
    # vectorizer = TfidfVectorizer()  # 限制特征数量
    # tfidf_matrix = vectorizer.fit_transform(texts)
    #
    # # 转换为密集矩阵
    # dense_matrix = tfidf_matrix.toarray()
    # print("TF-IDF 矩阵：\n", dense_matrix)
    # print(dense_matrix.shape)


    #
    # # 加载测试数据
    # newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    #
    # # 查看数据
    # print(newsgroups_train.data[0])  # 第一篇文档的内容
    # print(newsgroups_train.target[0])  # 第一篇文档的类别标签
    # print(newsgroups_train.target_names)