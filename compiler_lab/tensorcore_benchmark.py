"""Equal-precision Ampere tensor-core projection and cuBLAS controls."""

import argparse, json
from pathlib import Path
from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from .tensorcore import tensorcore_projection
from .benchmark import measure
from .metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    records = []
    for m, k, n in [(1, 32, 64), (33, 64, 128), (257, 128, 256), (1024, 128, 256)]:
        torch.manual_seed(1729)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        w = torch.randn(k, n, device="cuda", dtype=torch.float16) / k**0.5
        bias = torch.randn(n, device="cuda") * 0.1
        residual = torch.randn(m, n, device="cuda")
        out = torch.empty_like(residual)
        tensors = tuple(from_dlpack(t) for t in (x, w, bias, residual, out))
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = cute.compile(
            tensorcore_projection,
            *tensors,
            True,
            stream,
            options="--generate-line-info",
        )

        def cute_call():
            fn(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            return out

        def torch_call():
            return (torch.mm(x, w, out_dtype=torch.float32) + bias + residual).relu()

        expected = torch_call().clone()
        torch.testing.assert_close(cute_call(), expected, rtol=2e-4, atol=2e-4)
        funcs = {"cute": cute_call, "torch-cublas": torch_call}
        fixed_outputs = {"cute": out}
        refs = []
        for name in list(funcs):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    funcs[name]()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                result = funcs[name]()
            refs.extend([g, result])

            def replay(g=g, result=result):
                g.replay()
                return result

            funcs[name + "-graph"] = replay
            fixed_outputs[name + "-graph"] = result
        entry = {
            "shape": [m, k, n],
            "precision": "FP16 input/weights; FP32 accumulation, bias, residual, output",
            "engines": {},
        }
        for name, call in funcs.items():
            if name in fixed_outputs:
                fixed_outputs[name].fill_(float("nan"))
            torch.testing.assert_close(call(), expected, rtol=2e-4, atol=2e-4)
            ds = []
            hs = []
            for _ in range(5):
                d, h = measure(call, 200, 20)
                ds.append(d)
                hs.append(h)
            entry["engines"][name] = {
                "stream_summary": summarize([v for r in ds for v in r]),
                "host_summary": summarize([v for r in hs for v in r]),
                "stream_ms": ds,
                "host_ms": hs,
            }
        records.append(entry)
        (args.output / "report.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "cases": records,
                    "limits": "CUDA-event intervals may include dispatch idle gaps; repeated rounds in one process; original single-stage kernel, not a peak-performance claim",
                },
                indent=2,
            )
            + "\n"
        )
        print(
            [m, k, n],
            {k: v["stream_summary"]["p50_ms"] for k, v in entry["engines"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
