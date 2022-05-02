from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t
import ipdb
from main.models.resnet_autoencoder.model import VAE

from main.torch_model_modules.resnetmodel import ResNetAutoEncoder






class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = VAE(params)


    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self.generator(x)
        loss = self.loss(y_pred, y)
        return loss

    def test_step(self, batch, batch_idx:int):
        x, y = batch
        ipdb.set_trace()
        # shape of x is (batch_size, seq_len, stations, features)

        future_steps =t.ceil( y.shape[1]/ x.shape[1])
        outs = []

        for i in range(future_steps):
            out = self.generator(x)
            x = out
            outs.append(out)
        
        # stack outs in second dimension
        out = t.stack(outs, dim=1)
        out = out[:, -y.shape:, :, :]


        return self.test_without_forward(y, out)
