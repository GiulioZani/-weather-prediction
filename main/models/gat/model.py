from main.base_lightning_modules.base_model import BaseModel
from main.torch_model_modules.gat import GAT
from argparse import Namespace
import torch.nn.functional as F
import torch as t
import ipdb


class Model(BaseModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = GAT(params)

    def forward(self, x):
        acc = []
        xt = x
        for i in range(x.shape[1]):
            xt_hat = self.generator(xt)
            if i > x.shape[1] - 1:
                acc.append(xt_hat)
        result = t.stack(acc, dim=1)
        return result
