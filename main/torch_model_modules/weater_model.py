import torch as t
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):
        batch_size, seq_len, _ = x.size()
        # [batch_size, seq_len, seq_len]
        attention_weights = t.bmm(x, x.transpose(1, 2))
        # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention_weights, dim=1)
        # [batch_size, seq_len, seq_len]
        context = t.bmm(attention_weights, x)
        return context


class WeatherModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):

        return x
