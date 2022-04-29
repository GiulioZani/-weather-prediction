from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t
import ipdb

from .modules import EncoderDecoderConvLSTM




class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)


    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred, _ = self.generator(x)
        loss = self.loss(y_pred, y)
        return loss

    def test_step(self, batch, batch_idx:int):
        x, y = batch
        # ipdb.set_trace()
        # shape of x is (batch_size, seq_len, stations, features)

        out, _ = self.generator(x, future_step=y.shape[1])


        return self.test_without_forward(y, out)
