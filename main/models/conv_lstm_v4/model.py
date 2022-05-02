from ipaddress import ip_address
import math
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

    # def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
    #     x, y = batch
    #     y_pred, _ = self.generator(x)
    #     loss = self.loss(y_pred, y)
    #     return loss

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        # ipdb.set_trace()
        # shape of x is (batch_size, seq_len, stations, features)

        out = self.get_n_future_steps(x, y.shape[1])

        return self.test_without_forward(y, out)

    def validation_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ):
        x, y = batch
        # if batch_idx == 0:
        #     self.visualize_predictions(x, y, self(x), path=self.params.save_path)
            
        pred_y = self.get_n_future_steps(x, future_step=y.shape[1])
        loss = F.mse_loss(pred_y, y)
        self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss}

    def get_n_future_steps(self, x, future_step):

        seq = future_step
        future_steps = future_step
        future_steps = math.ceil(future_steps / x.shape[1])
        outs = []

        for i in range(future_steps):
            out = self.generator(x)
            x = out
            outs += [out]

        # ipdb.set_trace()
        out = t.stack(outs, dim=1)
        out = out.view(
            x.shape[0], int(future_steps * x.shape[1]), x.shape[2], x.shape[3]
        )
        out = out[:, -seq:, :, :]



        return out    