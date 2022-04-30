from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from ..utils.visualize_predictions import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os


class BaseModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        # self.data_manager = DataManger(data_path=params.data_location)
        self.generator = t.nn.Sequential()
        self.loss = t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        # if out is tuple, return the first element
        if isinstance(out, tuple):
            return out[0]
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    # def validation_epoch_end

    def validation_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)
        pred_y = self.generator(x, future_step=y.shape[1])[0]
        loss = F.mse_loss(pred_y, y)
        self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss}

    def test_without_forward(self, y, pred_y):
        y_single = y  # [:, :, -1, 2]
        pred_y_single = pred_y  # [:, :, -1, 2]
        se = F.mse_loss(pred_y_single, y_single, reduction="sum")
        denorm_pred_y = self.params.data_manager.denormalize(
            pred_y
        )  # , self.device)
        denorm_y = self.params.data_manager.denormalize(y)  # , self.device)
        ae = F.l1_loss(
            denorm_pred_y,
            denorm_y,
            reduction="sum",  # [:, :, -1, 2],  # [:, :, -1, 2],
        )
        temp_ae = F.l1_loss(
            denorm_pred_y[:, :, -1, 2],
            denorm_y[:, :, -1, 2],
            reduction="sum",  # [:, :, -1, 2],  # [:, :, -1, 2],
        )
        mask_pred_y = self.params.data_manager.discretize(
            denorm_pred_y
        )  # , self.device)
        mask_y = self.params.data_manager.discretize(denorm_y)  # [:, :, -1, 2]
        tn, fp, fn, tp = t.bincount(
            mask_y.flatten() * 2 + mask_pred_y.flatten(), minlength=4,
        )
        total_lengh = mask_y.numel()
        return {
            "se": se,
            "ae": ae,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "temp_ae": temp_ae,
            "total_lengh": total_lengh,
        }

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch

        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)

        pred_y = self(x)
        return self.test_without_forward(y, pred_y)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=self.params.lr)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.params.reduce_lr_on_plateau_patience,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mse",
        }

    def plot_test(self):
        test_dl = self.trainer.test_dataloaders[0]
        for batch in test_dl:
            x, y = batch
            pred_y = self.generator(x.to(self.device), future_step=y.shape[1])[
                0
            ]
            pred_y_city = pred_y[0, :, -1, 2].cpu()
            y_city = y[0, :, -1, 2]
            xs = t.arange(len(pred_y_city))
            plt.plot(xs, pred_y_city, label="prediction")
            plt.plot(xs, y_city, label="ground truth")
            plt.legend()
            plt.savefig(
                os.path.join(self.params.save_path, "168_final_prediction.png")
            )
            plt.clf()
            plt.plot(
                xs[: self.params.test_seq_len],
                pred_y_city[: self.params.test_seq_len],
                label="prediction",
            )
            plt.plot(
                xs[: self.params.test_seq_len],
                y_city[: self.params.test_seq_len],
                label="ground truth",
            )
            plt.legend()
            plt.savefig(
                os.path.join(self.params.save_path, "10_final_prediction.png")
            )

    def plot_importance_scores(self):
        if hasattr(self.generator, "get_importance_scores"):
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

    def test_epoch_end(self, outputs):
        self.plot_test()
        total_lenght = sum([x["total_lengh"] for x in outputs])
        mse = t.stack([x["se"] for x in outputs]).sum() / total_lenght
        mae = t.stack([x["ae"] for x in outputs]).sum() / total_lenght
        mae_temp = (
            t.stack([x["temp_ae"] for x in outputs]).sum() / total_lenght
        )
        tn = t.stack([x["tn"] for x in outputs]).sum() / total_lenght
        fp = t.stack([x["fp"] for x in outputs]).sum() / total_lenght
        fn = t.stack([x["fn"] for x in outputs]).sum() / total_lenght
        tp = t.stack([x["tp"] for x in outputs]).sum() / total_lenght
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        test_metrics = {
            "mse": mse,
            "mae": mae,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "mae_temp": mae_temp,
        }
        test_metrics = {k: v for k, v in test_metrics.items()}
        self.log("test_performance", test_metrics, prog_bar=True)
