import pytest
import torch
from compiler_lab.frontend import export_row_graph, UnsupportedGraph
from compiler_lab.mlp import MLPShape, make_mlp, ResidualMLP


def test_export_proves_row_local_graph():
    x, *weights = make_mlp(MLPShape(33, 32, 64))
    result = export_row_graph(ResidualMLP(*weights), x)
    assert result["shape"] == {"rows": 33, "channels": 32, "hidden": 64}
    assert "independent_rows" in result["mlir"]


def test_cross_row_reduction_is_rejected():
    class CrossRow(torch.nn.Module):
        def forward(self, x):
            return x + x.mean(dim=0)

    with pytest.raises(UnsupportedGraph):
        export_row_graph(CrossRow(), torch.randn(4, 8))


def test_wrong_precision_rejected_before_codegen():
    x, *weights = make_mlp(MLPShape(1, 8, 16))
    module = ResidualMLP(*(t.half() for t in weights))
    with pytest.raises(UnsupportedGraph):
        export_row_graph(module, x.half())
