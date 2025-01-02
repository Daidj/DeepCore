import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from macls.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling
# from macls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from deepcore.nets import EmbeddingRecorder


def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        padding = [kernel_size // 2, kernel_size // 2]

    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1
        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding


class Conv1d(nn.Module):

    def __init__(
            self,
            out_channels,
            kernel_size,
            in_channels,
            stride=1,
            dilation=1,
            padding='same',
            groups=1,
            bias=True,
            padding_mode='reflect', ):
        """_summary_

        Args:
            in_channels (int): intput channel or input data dimensions
            out_channels (int): output channel or output data dimensions
            kernel_size (int): kernel size of 1-d convolution
            stride (int, optional): strid in 1-d convolution . Defaults to 1.
            padding (str, optional): padding value. Defaults to "same".
            dilation (int, optional): dilation in 1-d convolution. Defaults to 1.
            groups (int, optional): groups in 1-d convolution. Defaults to 1.
            bias (bool, optional): bias in 1-d convolution . Defaults to True.
            padding_mode (str, optional): padding mode. Defaults to "reflect".
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)
        elif self.padding == 'causal':
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))
        elif self.padding == 'valid':
            pass
        else:
            raise ValueError(f"Padding must be 'same', 'valid' or 'causal'. Got {self.padding}")

        wx = self.conv(x)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)

        return x


class BatchNorm1d(nn.Module):
    def __init__(self, input_size, eps=1e-05, momentum=0.1, ):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.norm(x)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            activation=nn.ReLU,
            groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           dilation=dilation,
                           groups=groups)
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class AttentiveStatisticsPooling(nn.Module):
    """ASP
    This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)

        return pooled_stats


class TDNN(nn.Module):
    def __init__(self, num_class, input_size, channels=512, embd_dim=192):
        super(TDNN, self).__init__()
        self.emb_size = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=channels, dilation=1, kernel_size=5,
                                         stride=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.td_layer2 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=2, kernel_size=3,
                                         stride=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.td_layer3 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=3, kernel_size=3,
                                         stride=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.td_layer4 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1,
                                         stride=1)
        self.bn4 = nn.BatchNorm1d(channels)
        self.td_layer5 = torch.nn.Conv1d(in_channels=channels, out_channels=channels, dilation=1, kernel_size=1,
                                         stride=1)

        self.pooling = AttentiveStatisticsPooling(channels, 128)
        self.bn5 = nn.BatchNorm1d(channels * 2)
        self.linear = nn.Linear(channels * 2, embd_dim)
        self.bn6 = nn.BatchNorm1d(embd_dim)

        self.fc = nn.Linear(embd_dim, num_class)
        self.embedding_recorder = EmbeddingRecorder(False)

    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        out = self.embedding_recorder(out)
        out = self.fc(out)
        return out
