"""Original synthetic workloads; random weights do not establish model accuracy."""

import torch
from torch import nn


class ResidualDepthwiseBlock(nn.Module):
    """NCHW depthwise 3x3, pointwise 1x1, bias, residual and ReLU."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return torch.relu(self.pointwise(torch.relu(self.depthwise(x))) + x)


def make_case(channels=16, height=64, width=64, seed=1729):
    # Fork RNG state so making a case does not change a caller's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResidualDepthwiseBlock(channels).eval()
        x = torch.randn(1, channels, height, width)
    return model, x


class AttentionBlock(nn.Module):
    """Small static self-attention workload for layout/reduction coverage."""

    def __init__(self, dim=32):
        super().__init__()
        self.query=nn.Linear(dim,dim);self.key=nn.Linear(dim,dim);self.value=nn.Linear(dim,dim)
        self.scale=dim**-0.5

    def forward(self,x):
        q,k,v=self.query(x),self.key(x),self.value(x)
        scores=q@k.transpose(-1,-2)*self.scale
        return torch.softmax(scores,dim=-1)@v


class FixedGRU(nn.Module):
    """Three explicit recurrent steps; hidden state is an input and output."""

    def __init__(self,dim=16):
        super().__init__();self.cell=nn.GRUCell(dim,dim)

    def forward(self,inputs,hidden):
        for step in range(3):
            gi = torch.nn.functional.linear(inputs[:, step], self.cell.weight_ih, self.cell.bias_ih)
            gh = torch.nn.functional.linear(hidden, self.cell.weight_hh, self.cell.bias_hh)
            ir, iz, inn = gi.chunk(3, dim=-1)
            hr, hz, hn = gh.chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = (1 - update) * candidate + update * hidden
        return hidden
