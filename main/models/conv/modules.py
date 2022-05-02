import torch as t
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import ipdb


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size=(3, 3),
        padding="same",
        act="relu",
    ):
        super().__init__()
        # self.bn = nn.BatchNorm2d(in_channels)
        #self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        # self.ln = nn.LayerNorm([out_channels, 4, 5])
        self.do = nn.Dropout(0.5)
        self.act = nn.ReLU() if act == "relu" else nn.Sigmoid()
        self.net = nn.Sequential(self.conv, self.do, self.act,)

    def forward(self, x: t.Tensor) -> t.Tensor:
        out = self.net(x)  # self.relu(self.ln(self.do(self.conv(x))))
        return out + x[:, : out.shape[1]]


class Conv(nn.Module):
    def __init__(
        self, params: Namespace,
    ):
        super().__init__()
        self.params = params
        """
        self.net = nn.Sequential(
            nn.Conv2d(params.in_seq_len, 30, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(30, 20, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(20, 10, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(10, 5, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(5, 1, kernel_size=3, padding="same"),
            nn.Sigmoid(),
        )
        """
        self.net = nn.Sequential(
            # Block(params.in_seq_len, 100, padding="same"),
            # Block(100, 70, padding="same"),
            # Block(70, 50, padding="same"),
            # Block(50, 30, padding=0),
            # Block(30, 20, (2, 3), padding=0),
            Block(params.in_seq_len, params.in_seq_len),
            Block(params.in_seq_len, params.in_seq_len),
            Block(params.in_seq_len, params.in_seq_len),
            Block(params.in_seq_len, params.in_seq_len),
            Block(params.in_seq_len, params.in_seq_len),
            Block(params.in_seq_len, params.in_seq_len, act='sigmoid'),
            #Block(params.in_seq_len, params.in_seq_len),
            #Block(params.in_seq_len, params.in_seq_len),
        )

    def forward(self, x: t.Tensor, future_step=-1) -> t.Tensor:
        future_step = future_step if future_step > 0 else x.shape[1]
        context = x
        for _ in range(future_step // x.shape[1]):
            input = context[:, -self.params.in_seq_len :]
            next_step = self.net(input)  # .view(x.shape[0], 1, 4, 5)
            context = t.cat((context, next_step), dim=1)
        return context[:, -future_step:]
