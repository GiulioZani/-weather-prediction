from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

# from ..utils.visualize_predictions import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os


def to_var(x):
    return x[0, :, -1, 2].cpu()


class BaseModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        # self.data_manager = DataManger(data_path=params.data_location)
        self.generator = t.nn.Sequential()
        loss = t.nn.BCELoss()
        self.loss = lambda x, y: loss(x.flatten(), y.flatten()) # t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        # if out is tuple, return the first element
        if isinstance(out, tuple):
            return out[0]
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        context = x
        ipdb.set_trace()
        for _ in range(self.params.lag):
            pred = self.generator(context[:, : self.params.in_seq_len])
            context = t.cat((context, pred), dim=1)
        y_pred = context[:, -self.params.in_seq_len :]
        if batch_idx == 0:
            #self.plot_predictions(x, y, y_pred, "train", self.params.lag)
            pass
        # y_pred = self(x)out_seq_len
        # y = y[:, :, -1, 2]
        # y_pred = y_pred[:, :, -1, 2]
        ipdb.set_trace()
        loss = self.loss(y_pred.flatten(), y.flatten())
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_mse"] for x in outputs]).mean()
        self.log("val_mse", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        return {"val_mse": avg_loss}

    def predict(self, x, y):
        context = x[:, :self.params.in_seq_len]
        outputs = []
        for i in range(y.shape[1]):
            next_step = self.generator(context)
            next_context_step = x[:, self.params.in_seq_len + i].unsqueeze(0)
            next_context_step[0, 0, -1, 2] = next_step
            outputs.append(next_step)
            context = t.cat((context, next_context_step), dim=1)
            context = context[:, -self.params.in_seq_len:]
        y_pred = t.cat(outputs, dim=-1)
        return y_pred

    def validation_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ):
        x, y = batch
        y_pred = self.predict(x, y)
        if batch_idx == 0:
            # visualize_predictions(x, y, self(x), path=self.params.save_path)
            self.plot_predictions(x, y, y_pred, "val")
        loss = F.mse_loss(y_pred, y)
        return {"val_mse": loss}

    def configure_optimizers(self):
        optimizer = t.optim.Adam(
            self.parameters(), lr=self.params.lr, # weight_decay=0.01
        )
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

    def plot_predictions(
            self, x: t.Tensor, y: t.Tensor, pred_y: t.Tensor, title: str
    ):
        x = x[0]
        y = y[0]
        pred_y = pred_y[0]
        xs = t.arange(x.shape[0]) # + y.shape[1])
        """
        y = self.params.data_manager.denormalize(y.detach().cuda()).cpu()
        pred_y = self.params.data_manager.denormalize(
            pred_y.detach().cuda()
        ).cpu()
        x = self.params.data_manager.denormalize(x.detach().cuda()).cpu()
        """
        pred_y_city = pred_y.cpu() #pred_y[0, :, -1, 2].cpu()
        y_city = y.cpu() #y[0, :, -1, 2]
        x_city = x[:self.params.in_seq_len, -1, 2].cpu()
        plt.title(title)
        plt.plot(xs[: y.shape[0]], x_city, label="input")
        plt.plot(xs[y.shape[0]:], pred_y_city, label="prediction")
        plt.plot(xs[y.shape[0]:], y_city, label="ground truth")
        plt.legend()
        plt.savefig(os.path.join(self.params.save_path, f"{title}.png"))
        plt.close()
        """
        plt.plot(
            xs[: self.params.test_seq_len],
            pred_y_city[: self.params.test_seq_len],
            label="prediction",
        )
        plt.plot(
            xs[:10], y_city[:10], label="ground truth",
        )
        plt.legend()
        plt.savefig(
            os.path.join(self.params.save_path, "10_final_prediction.png")
        )
        """
    """
    def plot_test(self):
        test_dl = self.trainer.test_dataloaders[0]
        for batch in test_dl:
            x, y = batch
            y = y[:, : self.params.in_seq_len]
            pred_y = self.generator(x.to(self.device), future_step=y.shape[1])
            pred_y = pred_y[0] if isinstance(pred_y, tuple) else pred_y
            self.plot_predictions(x, y, pred_y, "test")
            break
    """

    def plot_importance_scores(self):
        importance_scores = self.generator.get_importance_scores()
        plt.title("Variable Importance Scores")
        plt.imshow(importance_scores[0], cmap="hot", interpolation="nearest")
        plt.savefig(
            os.path.join(
                self.params.save_path, "variables_importance_scores.png"
            )
        )
        plt.title("City Importance Scores")
        plt.imshow(importance_scores[1], cmap="hot", interpolation="nearest")
        plt.savefig(
            os.path.join(self.params.save_path, "city_importance_scores.png")
        )

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self.predict(x, y)
        if batch_idx == 0:
            self.plot_predictions(x, y, y_pred, "test")
            
        return self.test_without_forward(y, y_pred)

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
        """
        temp_ae = F.l1_loss(
            denorm_pred_y[:, :, -1, 2],
            denorm_y[:, :, -1, 2],
            reduction="sum",  # [:, :, -1, 2],  # [:, :, -1, 2],
        )
        """
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
            "total_length": total_lengh,
        }

    def test_epoch_end(self, outputs):
        total_lenght = sum([x["total_length"] for x in outputs])
        mse = t.stack([x["se"] for x in outputs]).sum() / total_lenght
        mae = t.stack([x["ae"] for x in outputs]).sum() / total_lenght
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
        }
        test_metrics = {k: v for k, v in test_metrics.items()}
        self.log("test_performance", test_metrics, prog_bar=True)
