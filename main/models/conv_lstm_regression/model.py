from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t

from main.models.conv_lstm_regression.modules import EncoderDecoderConvLSTM


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.params.lr)
