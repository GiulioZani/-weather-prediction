from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t

from .modules import EncoderDecoderConvLSTM


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)

    def test_step(self, batch, batch_idx:int):
        x, y = batch
        out, hidden = self.generator(x)
        acc = []
        for i in range(len(y.shape[1])):
            out, hidden = self.generator(out, hidden)
            acc.append(out)
        y_hat = t.stack(acc, dim=1)
        return self.test_without_forward(y, y_hat)

