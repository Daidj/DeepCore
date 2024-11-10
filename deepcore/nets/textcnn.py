# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://github.com/649453932/Chinese-Text-Classification-Pytorch

# class Config(object):
#
#     """配置参数"""
#     def __init__(self, dataset, embedding):
#         self.model_name = 'TextCNN'
#         self.train_path = dataset + '/data/train.txt'                                # 训练集
#         self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
#         self.test_path = dataset + '/data/test.txt'                                  # 测试集
#         self.class_list = [x.strip() for x in open(
#             dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
#         self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
#         self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
#         self.log_path = dataset + '/log/' + self.model_name
#         self.embedding_pretrained = torch.tensor(
#             np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
#             if embedding != 'random' else None                                       # 预训练词向量
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
#
#         self.dropout = 0.5                                              # 随机失活
#         self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = len(self.class_list)                         # 类别数
#         self.n_vocab = 0                                                # 词表大小，在运行时赋值
#         self.num_epochs = 20                                            # epoch数
#         self.batch_size = 128                                           # mini-batch大小
#         self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
#         self.learning_rate = 1e-3                                       # 学习率
#         self.embed = self.embedding_pretrained.size(1)\
#             if self.embedding_pretrained is not None else 300           # 字向量维度
#         self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
#         self.num_filters = 256                                          # 卷积核数量(channels数)
from deepcore.nets import EmbeddingRecorder

'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN(nn.Module):
    def __init__(self, num_classes, n_vocab, embed=300, num_filters=256, filter_sizes=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.embedding_recorder = EmbeddingRecorder(False)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.embedding_recorder(out)
        out = self.fc(out)
        return out
