import torch as t
from torch import nn
from torch.nn import functional as F
from ...base_lightning_modules.base_model import BaseModel
import ipdb


class OrthogonalMapping(nn.Module):
    def __init__(self, output_size: int=10):
        super().__init__()
        self.region_map = nn.Linear(5, 1, bias=False)
        self.feature_map = nn.Linear(4, 1, bias=False)
        self.to_out = nn.Linear(9, output_size)

    def single_forward(self, x: t.Tensor) -> t.Tensor:
        region_map = self.region_map(x).squeeze(-1)
        feature_map = self.feature_map(x.permute(0, 2, 1)).squeeze(-1)
        result = t.cat((region_map, feature_map), dim=1)
        result = F.relu(self.to_out(result))
        return result

    def forward(self, x: t.Tensor) -> t.Tensor:
        output = t.stack(
            tuple(
                self.single_forward(x[:, i]) for i in range(x.shape[1])
            ), dim=1
        )
        return output



class OrthogonalMappingReverse(nn.Module):
    def __init__(self, output_size: int=10):
        super().__init__()
        self.to_in = nn.Linear(output_size, 9)
        self.region_map = nn.Linear(4, 5, bias=False)
        self.feature_map = nn.Linear(5, 4, bias=False)

    def single_forward(self, x: t.Tensor) -> t.Tensor:
        x = F.relu(self.to_in(x))
        features = x[:,:5]
        regions = x[:,-4:]
        region_map = self.region_map(regions).unsqueeze(-1)
        feature_map = self.feature_map(features).unsqueeze(-1).permute(0, 2, 1)
        result = t.bmm(
            region_map, feature_map
        ).permute(0, 2, 1)
        return result

    def forward(self, x: t.Tensor) -> t.Tensor:
        output = t.stack(
            tuple(
                self.single_forward(x[:, i]) for i in range(x.shape[1])
            ), dim=1
        )
        return output

class VanillaLSTM(t.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = t.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.orthogonal_mapping = OrthogonalMapping(input_size)
        self.orthogonal_mapping_reverse = OrthogonalMappingReverse(input_size)
        self.out_act = t.nn.Sigmoid()
        # self.linear = t.nn.Linear(hidden_size, output_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        input = self.orthogonal_mapping(x)
        output, _ = self.lstm(input)
        output = self.orthogonal_mapping_reverse(output)
        output = self.out_act(output)
        return output

    def init_hidden(self):
        return t.autograd.Variable(t.zeros(1, 1, self.hidden_size))


class Model(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        # ipdb.set_trace()
        params.input_size = 20
        params.hidden_size = 20
        self.generator = VanillaLSTM(
            input_size=params.input_size, hidden_size=params.hidden_size,
        )
