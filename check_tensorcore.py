import json
from pathlib import Path
from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from compiler_lab.tensorcore import tensorcore_projection

results = []
for m, k, n in [(1, 7, 13), (17, 31, 47), (64, 64, 64), (257, 128, 256)]:
    torch.manual_seed(1729)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(k, n, device="cuda", dtype=torch.float16) / k**0.5
    b = torch.randn(n, device="cuda") * 0.1
    r = torch.randn(m, n, device="cuda")
    out = torch.empty(m, n, device="cuda")
    args = tuple(from_dlpack(t) for t in (x, w, b, r, out))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(
        tensorcore_projection, *args, True, stream, options="--generate-line-info"
    )
    print("CASE", m, k, n, flush=True)
    fn(*args, stream)
    torch.cuda.synchronize()
    expected = (x.float() @ w.float() + b + r).relu()
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-4)
    error = (out - expected).abs().max().item()
    results.append({"shape": [m, k, n], "max_abs_error": error, "passed": True})
Path("runs/tensorcore-checks").mkdir(exist_ok=True)
Path("runs/tensorcore-checks/report.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print("TENSORCORE_PASS", flush=True)
