"""Prove the supported FX graph before emitting row-local scheduling IR."""

from dataclasses import asdict
import hashlib
import json
import re
import subprocess
import torch
from .mlp import MLPShape, ResidualMLP


class UnsupportedGraph(ValueError):
    pass


def export_row_graph(module, x):
    program = torch.export.export(module, (x,))
    graph = program.graph_module.graph
    calls = [n for n in graph.nodes if n.op == "call_function"]
    names = [str(n.target) for n in calls]
    expected = [
        "aten.matmul.default",
        "aten.add.Tensor",
        "aten.relu.default",
        "aten.matmul.default",
        "aten.add.Tensor",
        "aten.add.Tensor",
        "aten.relu.default",
    ]
    if names != expected or any(n.kwargs.get("alpha", 1) != 1 for n in calls):
        raise UnsupportedGraph("requires the exact two-projection/ReLU/residual graph")
    specs = program.graph_signature.input_specs
    user = [s for s in specs if s.kind.name == "USER_INPUT"]
    if len(user) != 1:
        raise UnsupportedGraph("one dynamic input required")
    placeholders = {n.name: n for n in graph.nodes if n.op == "placeholder"}
    inp = placeholders[user[0].arg.name]
    buffers = {
        placeholders[s.arg.name]: program.state_dict[s.target]
        for s in specs
        if s.kind.name in ("BUFFER", "PARAMETER")
    }
    a0, a1, a2, a3, a4, a5, a6 = calls
    # The connectivity proves every row can be computed without another CTA.
    if (
        a0.args[0] != inp
        or a1.args[0] != a0
        or a2.args[0] != a1
        or a3.args[0] != a2
        or a4.args[0] != a3
        or a5.args[:2] != (a4, inp)
        or a6.args[0] != a5
    ):
        raise UnsupportedGraph("unsupported connectivity or cross-row dependency")
    try:
        weights = tuple(
            buffers[n] for n in (a0.args[1], a1.args[1], a3.args[1], a4.args[1])
        )
    except KeyError as e:
        raise UnsupportedGraph("weights and biases must be frozen") from e
    w1, b1, w2, b2 = weights
    if x.ndim != 2 or any(t.dtype != torch.float32 for t in (x, *weights)):
        raise UnsupportedGraph("two-dimensional float32 input required")
    m, c = x.shape
    h = w1.shape[1] if w1.ndim == 2 else 0
    if (
        tuple(w1.shape) != (c, h)
        or tuple(b1.shape) != (h,)
        or tuple(w2.shape) != (h, c)
        or tuple(b2.shape) != (c,)
    ):
        raise UnsupportedGraph("unsupported weight shapes")
    output = next(n for n in graph.nodes if n.op == "output")
    if output.args[0] != (a6,):
        raise UnsupportedGraph("single final output required")
    shape = MLPShape(m, c, h)
    signature = hashlib.sha256(str(graph).encode()).hexdigest()
    ir = (
        f'module {{\n  "lab.graph"() {{kind = "residual_mlp", dtype = "f32", dependency_scope = "independent_rows", '
        f'rows = {m} : i64, channels = {c} : i64, hidden = {h} : i64, fx_sha256 = "{signature}"}} : () -> ()\n}}\n'
    )
    return {
        "shape": asdict(shape),
        "weights": weights,
        "fx_graph": str(graph),
        "mlir": ir,
        "fx_sha256": signature,
    }


def compile_schedule(exported, iree_opt, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "graph.mlir").write_text(exported["mlir"])
    (output_dir / "graph.fx.txt").write_text(exported["fx_graph"])
    cmd = [
        str(iree_opt),
        "--iree-plugin=lab",
        "--lab-schedule-row-mlp",
        str(output_dir / "graph.mlir"),
        "-o",
        str(output_dir / "schedule.mlir"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (output_dir / "compile.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    ir = (output_dir / "schedule.mlir").read_text()
    strategy = re.search(r'strategy\s*=\s*"(persistent|conventional)"', ir)
    workers = re.search(r"workers\s*=\s*(\d+)\s*:\s*i64", ir)
    if not strategy or not workers:
        raise RuntimeError("compiler did not return a supported schedule")
    result = {
        "strategy": strategy.group(1),
        "workers": int(workers.group(1)),
        "shape": exported["shape"],
        "fx_sha256": exported["fx_sha256"],
        "compiler_command": cmd,
    }
    (output_dir / "schedule.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
