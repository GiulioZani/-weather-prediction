from ipaddress import ip_address
from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t
import ipdb
from .modules import Conv


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = Conv(params)

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y = y[:, -1]  # we only care about last frame
        y_pred = self.generator(x)
        loss = self.loss(y_pred.squeeze(), y)
        return loss

