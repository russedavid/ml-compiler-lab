"""Typed graph contract and PyTorch reference for the persistent-kernel study."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MLPShape:
    rows: int
    channels: int
    hidden: int

    def validate(self):
        if min(self.rows, self.channels, self.hidden) < 1:
            raise ValueError("all dimensions must be positive")
        if max(self.channels, self.hidden) > 256:
            raise ValueError("the row-owned kernel supports channels/hidden <= 256")


def make_mlp(shape: MLPShape, seed=1729):
    shape.validate()
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape.rows, shape.channels, generator=g)
    # Small weight scale avoids unrepresentative explosive activations.
    w1 = torch.randn(shape.channels, shape.hidden, generator=g) / shape.channels**0.5
    b1 = torch.randn(shape.hidden, generator=g) * 0.1
    w2 = torch.randn(shape.hidden, shape.channels, generator=g) / shape.hidden**0.5
    b2 = torch.randn(shape.channels, generator=g) * 0.1
    return x, w1, b1, w2, b2


def reference_mlp(x, w1, b1, w2, b2):
    return ((x @ w1 + b1).relu() @ w2 + b2 + x).relu()


class ResidualMLP(torch.nn.Module):
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        for name, value in zip(("w1", "b1", "w2", "b2"), (w1, b1, w2, b2)):
            self.register_buffer(name, value)

    def forward(self, x):
        return ((x @ self.w1 + self.b1).relu() @ self.w2 + self.b2 + x).relu()
