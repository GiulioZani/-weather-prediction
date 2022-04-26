import h5py
import torch as t
import ipdb


class DataManager:
    def __init__(self, *, data: t.Tensor, norm_type="minmax"):
        self.norm_type = norm_type
        flattened_data = data.reshape(-1, 5).float()
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.min = flattened_data.min(dim=0)[0][None, None, :].to(device)
        self.max = flattened_data.max(dim=0)[0][None, None, :].to(device)
        self.var = t.var(flattened_data, dim=0)[None, None, :].to(device)
        self.mean = t.mean(flattened_data, dim=0)[None, None, :].to(device)

    def normalize(self, data: t.Tensor) -> t.Tensor:
        old_device = self.min.device
        if data.device != self.min.device:
            old_device = data.device
            data = data.to(self.min.device)
        result = data
        if self.norm_type == "minmax":
            result = (data - self.min) / (self.max - self.min)
        elif type == "meanvar":
            result = (data - self.mean) / self.var
        result = result.to(old_device)
        return result

    def denormalize(self, data: t.Tensor) -> t.Tensor:
        result = data
        if self.norm_type == "minmax":
            result = data * (self.max - self.min) + self.min
        elif type == "meanvar":
            result = data * self.var + self.mean
        return result

    def discretize(self, data: t.Tensor) -> t.Tensor:
        return data > self.mean
