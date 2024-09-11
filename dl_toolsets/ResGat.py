
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn import Conv1d
class OneDimConvBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=2048):
        super().__init__()
        self.attention_conv = OneDimAttention(in_channel, in_channel)
        self.batchnorm1 = torch.nn.BatchNorm1d(in_channel)
        self.batchnorm2 = torch.nn.BatchNorm1d(in_channel)
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.linear2 = nn.Linear(in_channel, out_channel)
        self.ffn = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.attention_conv(x, x, x)
        h = self.batchnorm1(x + h)

        h_new = self.ffn(h)
        h_new = self.batchnorm2(h + h_new)
        return F.dropout1d(self.linear2(h_new), training=self.training)


class OneDimAttention(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = torch.tensor(in_size)
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, q, k, v):
        attention = torch.mul(q, k) / torch.sqrt(self.in_size)
        attention = self.linear(attention)
        return torch.mul(F.softmax(attention, dim=-1), v)