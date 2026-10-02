import json
from pathlib import Path
from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from compiler_lab.mlp import MLPShape, make_mlp
from compiler_lab.tensorcore_mlp import persistent_tensorcore_mlp


def reference(x, w1, b1, w2, b2):
    hidden = (torch.mm(x, w1, out_dtype=torch.float32) + b1).relu().half()
    return (torch.mm(hidden, w2, out_dtype=torch.float32) + b2 + x.float()).relu()


torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
records = []
for shape in [
    MLPShape(1, 7, 13),
    MLPShape(33, 32, 64),
    MLPShape(129, 31, 47),
    MLPShape(257, 64, 128),
    MLPShape(1024, 128, 256),
]:
    values = make_mlp(shape)
    x, w1, b1, w2, b2 = (v.cuda() for v in values)
    x = x.half()
    w1 = w1.half()
    w2 = w2.half()
    out = torch.empty_like(x, dtype=torch.float32)
    workers = min(32, (shape.rows + 15) // 16)
    args = tuple(from_dlpack(t) for t in (x, w1, b1, w2, b2, out))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    print("CASE", shape, flush=True)
    fn = cute.compile(
        persistent_tensorcore_mlp,
        *args,
        workers,
        stream,
        options="--generate-line-info",
    )
    fn(*args, workers, stream)
    torch.cuda.synchronize()
    expected = reference(x, w1, b1, w2, b2)
    torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-4)
    records.append(
        {
            "shape": shape.__dict__,
            "workers": workers,
            "max_abs_error": float((out - expected).abs().max()),
            "passed": True,
        }
    )
Path("runs/tensorcore-mlp-checks").mkdir(exist_ok=True)
Path("runs/tensorcore-mlp-checks/report.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
print("TENSORCORE_MLP_PASS", flush=True)
