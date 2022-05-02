from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t
from .modules import EncoderDecoderEmbeddings
import ipdb


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = EncoderDecoderEmbeddings(params)


    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        # y = y[:, -1]  # we only care about last frame
        y_pred = self.generator(x, future_step=y.shape[1])
        loss = self.loss(y_pred.squeeze(), y)
        return loss



    def test_step(self, batch, batch_idx:int):
        x, y = batch
        # shape of x is (batch_size, seq_len, stations, features)

       
        out = self.generator(x, future_step=y.shape[1])
        # ipdb.set_trace()

        return self.test_without_forward(y, out)