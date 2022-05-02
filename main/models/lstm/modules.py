import torch as t
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import ipdb


class LSTM(nn.Module):
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
        self.layers = 1
        self.hidden = params.in_seq_len * params.in_seq_len
        # self.lin1 = nn.Linear(20, self.hidden, bias=False)
        self.lin1 = nn.Sequential(
            nn.Conv2d(
                1,
                params.in_seq_len * params.in_seq_len,
                kernel_size=3,
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(
                params.in_seq_len * params.in_seq_len,
                params.in_seq_len * params.in_seq_len,
                kernel_size=(2, 3),
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.net = nn.LSTM(
            self.hidden, self.hidden, self.layers, batch_first=True,
        )
        self.lin2 = nn.Linear(self.hidden, 20, bias=False)

    def forward(self, x: t.Tensor, future_step=0) -> t.Tensor:
        # initialize the hidden state.
        # x_in = t.relu(self.lin1(x.view(x.shape[0], x.shape[1], 20)))
        x_in = t.stack(
            tuple(
                self.lin1(x[:, i].unsqueeze(1)) for i in range(x.shape[1])
            ),
            dim=1
        ).view(x.shape[0], x.shape[1], -1)
        hidden = (
            t.randn(self.layers, x.shape[0], self.hidden).to(x.device),
            t.randn(self.layers, x.shape[0], self.hidden).to(x.device),
        )
        """
        future_step = future_step if future_step > 0 else x.shape[1]
        context = x
        for _ in range(future_step // x.shape[1]):
            input = context[:, -self.params.in_seq_len :]
            next_step = self.net(input)  # .view(x.shape[0], 1, 4, 5)
            context = t.cat((context, next_step), dim=1)
        return context[:, -future_step:]
        """
        output, hidden = self.net(x_in, hidden)
        last_output = output[:, -1, :].unsqueeze(1)
        for _ in range(future_step):
            last_output, hidden = self.net(last_output, hidden)
            output = t.cat((output, last_output), dim=1)
        output = output[:, -future_step:, :]
        return t.sigmoid(
            self.lin2(output).view(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            )
        )
