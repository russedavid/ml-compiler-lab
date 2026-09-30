import json
from pathlib import Path
import torch
from compiler_lab.mlp import MLPShape, make_mlp, reference_mlp
from compiler_lab.runner import MLPExecutor
from compiler_lab.metrics import compare

torch.backends.cuda.matmul.allow_tf32 = False
results = []
for shape in [MLPShape(1, 7, 13), MLPShape(33, 32, 64), MLPShape(257, 64, 128)]:
    values = make_mlp(shape)
    expected = reference_mlp(*values).numpy()
    for strategy in ["separate", "fused", "persistent", "persistent-global"]:
        print(shape, strategy, flush=True)
        runner = MLPExecutor(values, strategy)
        actual = runner().cpu().numpy()
        check = compare(actual, expected)
        print(check, flush=True)
        assert check["passed"]
        # Changed requests and nondefault stream check stale state and stream use.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            runner.tensors[0].neg_()
            actual = runner().cpu().numpy()
        expected2 = reference_mlp(-values[0], *values[1:]).numpy()
        check2 = compare(actual, expected2)
        assert check2["passed"], check2
        results.append(
            {
                "shape": shape.__dict__,
                "strategy": strategy,
                "correctness": check,
                "changed_input": check2,
            }
        )
Path("runs/kernel-checks").mkdir(exist_ok=True)
Path("runs/kernel-checks/report.json").write_text(json.dumps(results, indent=2) + "\n")
print("ALL_KERNELS_PASS", flush=True)
