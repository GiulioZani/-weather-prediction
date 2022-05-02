import torch as t
import torch.nn as nn
import torch.nn.functional as F


class AxialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            (4, 1),
            stride,
            "same",
            dilation,
            groups,
            bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            (5, 1),
            stride,
            "same",
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        return F.relu(x_1 + x_2)

    def get_attention_scores(self):
        res = self.conv1.weight.detach().cpu().abs().mean(dim=1, keepdim=True), self.conv2.weight.detach().cpu().abs().mean(dim=1, keepdim=True)
        return res
