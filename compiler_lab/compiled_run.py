"""Verified FX graph -> C++ MLIR schedule -> numerical GPU execution."""

import argparse, json
from pathlib import Path
import torch
from .frontend import export_row_graph, compile_schedule
from .mlp import MLPShape, make_mlp, ResidualMLP, reference_mlp
from .runner import MLPExecutor
from .metrics import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iree-opt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=33)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()
    shape = MLPShape(args.rows, args.channels, args.hidden)
    values = make_mlp(shape)
    module = ResidualMLP(*values[1:])
    exported = export_row_graph(module, values[0])
    schedule = compile_schedule(exported, args.iree_opt, args.output)
    if schedule["strategy"] == "persistent":
        runner = MLPExecutor(values, "persistent", workers=schedule["workers"])
        actual = runner().cpu().numpy()
    else:
        device = tuple(t.cuda() for t in values)
        actual = reference_mlp(*device).cpu().numpy()
    expected = reference_mlp(*values).numpy()
    result = compare(actual, expected)
    (args.output / "correctness.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise AssertionError(result)
    print(json.dumps({"schedule": schedule, "correctness": result}, indent=2))


if __name__ == "__main__":
    main()
