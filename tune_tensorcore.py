import argparse, json
from pathlib import Path
from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from compiler_lab.tensorcore import tensorcore_projection
from compiler_lab.benchmark import measure
from compiler_lab.metrics import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
options.output.mkdir(parents=True, exist_ok=False)
results = []
for m, k, n in [(33, 64, 128), (257, 128, 256)]:
    torch.manual_seed(1729)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(k, n, device="cuda", dtype=torch.float16) / k**0.5
    b = torch.zeros(n, device="cuda")
    r = torch.randn(m, n, device="cuda")
    out = torch.empty_like(r)
    args = tuple(from_dlpack(t) for t in (x, w, b, r, out))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    expected = (torch.mm(x, w, out_dtype=torch.float32) + b + r).relu()
    for tm, tn in [(16, 8), (32, 16), (32, 32), (64, 32)]:
        print("TILE", m, k, n, tm, tn, flush=True)
        fn = cute.compile(tensorcore_projection, *args, True, stream, tm, tn)
        fn(*args, stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-4)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-4)
        d, h = measure(g.replay, 200, 20)
        results.append(
            {
                "shape": [m, k, n],
                "tile": [tm, tn],
                "correctness": True,
                "graph_stream_summary": summarize(d),
                "host_summary": summarize(h),
            }
        )
(options.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
print("TUNING_PASS", flush=True)
