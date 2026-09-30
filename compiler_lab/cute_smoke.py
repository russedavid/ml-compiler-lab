"""Original residual/ReLU kernel: an installation and tail-predication check.

This is an epilogue kernel, not a graph-level megakernel or a tuned GEMM.
"""

import json
from pathlib import Path
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def residual_relu_kernel(x: cute.Tensor, residual: cute.Tensor, out: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < cute.size(x):
        value = x[i] + residual[i]
        if value > cutlass.Float32(0):
            out[i] = value
        else:
            out[i] = cutlass.Float32(0)


@cute.jit
def residual_relu(x: cute.Tensor, residual: cute.Tensor, out: cute.Tensor):
    residual_relu_kernel(x, residual, out).launch(
        grid=(cute.ceil_div(cute.size(x), 256), 1, 1), block=(256, 1, 1)
    )


def main():
    torch.manual_seed(1729)
    results = []
    for size in [1, 31, 256, 257, 65537]:
        x = torch.randn(size, device="cuda")
        r = torch.randn_like(x)
        out = torch.full_like(x, float("nan"))
        args = tuple(from_dlpack(t) for t in (x, r, out))
        fn = cute.compile(residual_relu, *args)
        fn(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, (x + r).relu(), rtol=0, atol=0)
        # A second request with changed data must not reuse an old output.
        x.neg_()
        out.fill_(float("nan"))
        fn(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, (x + r).relu(), rtol=0, atol=0)
        results.append({"elements": size, "passed": True, "requests": 2})
    result = {
        "purpose": "CuTe execution and predication smoke, not a performance benchmark",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cases": results,
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/cute-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
