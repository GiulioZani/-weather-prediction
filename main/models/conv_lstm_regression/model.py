from main.base_lightning_modules.base_model import BaseModel
from argparse import ArgumentParser
import torch.nn.functional as F
import torch as t

from .modules import EncoderDecoderConvLSTM


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)

    def plot_importance_scores(self):
        importance_scores = self.generator.get_importance_scores()
        plt.title("Variable Importance Scores")
        plt.imshow(
            importance_scores[0], cmap="hot", interpolation="nearest"
        )
        plt.savefig(
            os.path.join(
                self.params.save_path, "variables_importance_scores.png"
            )
        )
        plt.title("City Importance Scores")
        plt.imshow(
            importance_scores[1], cmap="hot", interpolation="nearest"
        )
        plt.savefig(
            os.path.join(
                self.params.save_path, "city_importance_scores.png"
            )
        )


    def test_step(self, batch, batch_idx:int):
        x, y = batch
        out, hidden = self.generator(x)
        acc = []
        for i in range(len(y.shape[1])):
            out, hidden = self.generator(out, hidden)
            acc.append(out)
        y_hat = t.stack(acc, dim=1)
        return self.test_without_forward(y, y_hat)

