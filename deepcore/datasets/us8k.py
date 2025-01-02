import os
from typing import Any

import gensim
import pyarrow.parquet as pq
import nltk
import pickle
import torch
import numpy as np
import torchaudio
from gensim import downloader
from torch.utils.data import Dataset, DataLoader
from gensim.models import word2vec
import matplotlib.pyplot as plt
import soundata
from yeaudio.audio import AudioSegment
from torchaudio.transforms import MFCC


class UrbanSound8KDataSet(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.root_dir = root
        self.train = train
        self.data: Any = []
        self.targets = []
        self.classes = [x for x in range(10)]
        self.max_duration = 4.0
        self.input_size = None
        self.init_dataset()
        self.num_classes = len(self.classes)  # 类别数


        print("UrbanSound8K Dataset init")

    def __getitem__(self, index):
        features, target = self.data[index], self.targets[index]
        # text = self.data[item][:self.max_len]
        # text_idx = [self.word_2_index.get(i, 0) for i in text]
        # text_idx = text_idx + [1] * (self.max_len - len(text))
        # label = int(self.targets[item])
        return features, target

    def __len__(self):
        return len(self.data)

    def init_dataset(self):
        folder_name = self.root_dir

        if self.train:
            self.data = torch.load(os.path.join(folder_name, 'train_samples.pt'))
            self.targets = torch.load(os.path.join(folder_name, 'train_labels.pt'))

        else:
            self.data = torch.load(os.path.join(folder_name, 'test_samples.pt'))
            self.targets = torch.load(os.path.join(folder_name, 'test_labels.pt'))
        self.input_size = self.data[0].size(1)

def build_dataset(data_home):
    dataset = soundata.initialize('urbansound8k', data_home=data_home)
    target_sample_rate = 16000
    target_db = -20
    final_duration = 4.0
    # dataset.download()  # download the dataset
    # dataset.validate()  # validate that all the expected files are there
    # 加载数据
    data = dataset.load_clips()
    input_samples = []
    targets = []
    max_duration = 0.0
    for clip in data.values():
        audio_path = clip.audio_path
        label = clip.class_id
        # label_name = clip.class_label
        audio_segment = AudioSegment.from_file(audio_path)
        # 重采样
        if audio_segment.sample_rate != target_sample_rate:
            audio_segment.resample(target_sample_rate)
        # 音量归一化
        audio_segment.normalize(target_db=target_db)
        # 裁剪需要的数据
        if audio_segment.duration > max_duration:
            max_duration = audio_segment.duration
        while audio_segment.duration > final_duration:
            audio_segment.crop(duration=final_duration)
        while audio_segment.duration < final_duration:
            silence = round(final_duration, 4) - round(audio_segment.duration, 4)
            silence = max(0.0001, silence)
            audio_segment.pad_silence(silence, sides='end')

        samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
        mfcc = MFCC(sample_rate=target_sample_rate)
        feature = mfcc(samples)
        feature = feature.squeeze(0)
        label = torch.tensor(int(label), dtype=torch.int64)
        input_samples.append(feature)
        targets.append(label)
        # if len(input_samples) > 23:
        #     break

    print(max_duration)
    all_samples = torch.stack(input_samples)
    all_targets = torch.stack(targets)
    print(all_targets)
    size = len(all_targets)
    train_num = round(5*size/6)
    torch.save(all_samples[0:train_num], os.path.join(data_home, 'train_samples.pt'))
    torch.save(all_targets[0:train_num], os.path.join(data_home, 'train_labels.pt'))
    torch.save(all_samples[train_num:], os.path.join(data_home, 'test_samples.pt'))
    torch.save(all_targets[train_num:], os.path.join(data_home, 'test_labels.pt'))

    return all_samples, all_targets

def UrbanSound8K(data_path):
    channel = None

    mean = None
    std = None
    folder_name = os.path.join(data_path, 'UrbanSound8K')
    if not os.path.exists(os.path.join(folder_name, 'train_samples.pt')):
        build_dataset(folder_name)
        print("Features file built successfully!")

    dst_train = UrbanSound8KDataSet(root=folder_name, train=True)
    num_classes = dst_train.num_classes
    class_names = dst_train.classes
    dst_test = UrbanSound8KDataSet(root=folder_name, train=False)
    im_size = dst_train.data[0].size(1)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

if __name__ == '__main__':
    data_home = '/home/sample_selection/data/'
    UrbanSound8K(data_home)
    # x, y = build_dataset(data_home)
    # train_smaples = torch.load(os.path.join(data_home, 'train_samples.pt'))
    # print(len(train_smaples))
    # print(train_smaples[0])
    #
    # train_labels = torch.load(os.path.join(data_home, 'train_labels.pt'))
    # print(train_labels[0])
    # test_samples = torch.load(os.path.join(data_home, 'test_samples.pt'))
    # test_labels = torch.load(os.path.join(data_home, 'test_labels.pt'))
    exit(0)
    # UrbanSound8K('/home/sample_selection/data')
    dataset = soundata.initialize('urbansound8k', data_home='/home/sample_selection/data/UrbanSound8K')
    # dataset.download()  # download the dataset
    # dataset.validate()  # validate that all the expected files are there

    # 加载数据
    data = dataset.load_clips()
    input_samples = []
    targets = []
    # 打印数据集中的音频文件路径
    i = 0
    max_duration = 0.0
    for clip in data.values():
        # if i != 82:
        #     i += 1
        #     continue
        print(clip.audio_path)
        audio_path = clip.audio_path
        label = clip.class_id
        label_name = clip.class_label
        # audio = clip.audio
    # 读取音频
        audio_segment = AudioSegment.from_file(audio_path)
        # waveform, sample_rate = torchaudio.load(audio_path)

        target_sample_rate = 16000
        # 重采样
        if audio_segment.sample_rate != target_sample_rate:
            audio_segment.resample(target_sample_rate)
        # 音量归一化
        audio_segment.normalize(target_db=-20)
        # 裁剪需要的数据
        final_duration = 4.0
        mode = 'train'
        if audio_segment.duration > max_duration:
            max_duration = audio_segment.duration
        while audio_segment.duration > final_duration:
            audio_segment.crop(duration=final_duration)
        while audio_segment.duration < final_duration:
            silence = round(final_duration, 4) - round(audio_segment.duration, 4)
            silence = max(0.0001, silence)
            audio_segment.pad_silence(silence, sides='end')


        samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
        mfcc = MFCC(sample_rate=target_sample_rate)
        feature = mfcc(samples)
        feature = feature.squeeze(0)
        label = torch.tensor(int(label), dtype=torch.int64)
        input_samples.append(feature)
        targets.append(label)

    print(max_duration)
    all_sample = torch.stack(input_samples)
    all_targets = torch.stack(targets)
    print(all_targets)
