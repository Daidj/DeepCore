import huggingface_hub
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from deepcore.datasets import YELP, AGNews
from deepcore.datasets.sst2 import SST2
from deepcore.datasets.sst5 import SST5
from deepcore.datasets.thucnews import THUCNews
from deepcore.datasets.us8k import UrbanSound8K
from deepcore.nets.tdnn import TDNN
from deepcore.nets.textcnn import TextCNN

if __name__ == '__main__':

    channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset = UrbanSound8K('./data')

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    # out_channel = num_filter
    epochs = 200
    lr = 1e-3
    input_size = train_dataset.data[0].size(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = TDNN(num_class=num_classes, input_size=input_size).to(device)
    print(network.parameters)
    optimizer = torch.optim.Adam(network.parameters(), lr)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    total_step = len(train_loader)
    for epoch in range(epochs):
        network.train()

        for i, contents in enumerate(train_loader):
            optimizer.zero_grad()

            target = contents[1].to(device)
            input = contents[0].to(device)
            # print(input.shape)
            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

            # Measure accuracy and record loss
            # prec1 = accuracy(output.data, target, topk=(1,))[0]
            # losses.update(loss.data.item(), input.size(0))
            # top1.update(prec1.item(), input.size(0))

            # Compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            if i % 2 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    # for batch_idx, batch_label in enumerate(test_loader):
    #     pre = network(batch_idx.to(device))
    #     batch_label = batch_label.to(device)
    #     right_num += torch.sum(pre == batch_label)
    # print(f'acc = {right_num / len(dev_text) * 100:.3f}%')
