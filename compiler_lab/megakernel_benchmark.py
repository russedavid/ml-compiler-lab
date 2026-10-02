"""Equal-precision complete residual-MLP comparison with replay validation."""

import argparse, json
import hashlib
import importlib.metadata as metadata
import subprocess
from pathlib import Path
import torch
from .mlp import MLPShape, make_mlp
from .tensorcore_runner import TensorCoreMLPExecutor, tensorcore_mlp_reference
from .benchmark import measure
from .metrics import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assessment", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    shapes = [
        MLPShape(1, 7, 13),
        MLPShape(33, 32, 64),
        MLPShape(129, 31, 47),
        MLPShape(257, 64, 128),
        MLPShape(1024, 128, 256),
    ]
    if args.assessment:
        shapes = [
            MLPShape(5, 19, 37),
            MLPShape(49, 48, 96),
            MLPShape(193, 63, 111),
            MLPShape(513, 96, 192),
        ]
    report = {
        "precision": "FP16 input/weights and hidden state; FP32 accumulation, bias and output",
        "seed": 2718 if args.assessment else 1729,
        "rounds": 5, "samples_per_round": 200,
        "packages": {name: metadata.version(name) for name in ("torch", "nvidia-cutlass-dsl")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
        "device": torch.cuda.get_device_name(),
        "assessment": args.assessment,
        "limits": "CUDA-event stream intervals include possible dispatch idle gaps; rounds share one process. Original bounded MLP, not whole LLM inference.",
        "cases": [],
    }
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,memory.used", "--format=csv"], text=True)
    (args.output / "hardware.csv").write_text(hardware)
    for shape in shapes:
        values = make_mlp(shape, seed=2718 if args.assessment else 1729)
        runners = {
            strategy: TensorCoreMLPExecutor(values, strategy)
            for strategy in ("conventional", "persistent", "persistent-global")
        }
        base = runners["conventional"].tensors

        def eager():
            return tensorcore_mlp_reference(*base)

        expected = eager().clone()
        funcs = {**runners, "torch-cublas": eager}
        fixed_outputs = {name: runner.output for name, runner in runners.items()}
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
                output = funcs[name]()
            refs.extend([g, output])

            def replay(g=g, output=output):
                g.replay()
                return output

            funcs[name + "-graph"] = replay
            fixed_outputs[name + "-graph"] = output
        entry = {"shape": shape.__dict__, "engines": {}}
        for name, fn in funcs.items():
            if name in fixed_outputs:
                fixed_outputs[name].fill_(float("nan"))
            torch.testing.assert_close(fn(), expected, rtol=3e-4, atol=3e-4)
            entry["engines"][name] = {"passed": True, "stream_ms": [], "host_ms": []}
        names = list(funcs)
        for round_no in range(5):
            order = names[round_no % len(names) :] + names[: round_no % len(names)]
            for name in order:
                d, h = measure(funcs[name], 200, 20)
                entry["engines"][name]["stream_ms"].append(d)
                entry["engines"][name]["host_ms"].append(h)
        for runner in runners.values():
            runner.tensors[0].neg_()
            runner.output.fill_(float("nan"))
            runner.hidden.fill_(float("nan"))
            runner.hidden_half.fill_(float("nan"))
            if hasattr(runner, "global_hidden"):
                runner.global_hidden.fill_(float("nan"))
        changed = eager().clone()
        for name, fn in funcs.items():
            if name in fixed_outputs:
                fixed_outputs[name].fill_(float("nan"))
            torch.testing.assert_close(fn(), changed, rtol=3e-4, atol=3e-4)
            result = entry["engines"][name]
            result["replay_changed_input"] = True
            result["stream_summary"] = summarize(
                [v for r in result["stream_ms"] for v in r]
            )
            result["host_summary"] = summarize(
                [v for r in result["host_ms"] for v in r]
            )
        report["cases"].append(entry)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            shape,
            {
                k: round(v["stream_summary"]["p50_ms"] * 1000, 2)
                for k, v in entry["engines"].items()
            },
            flush=True,
        )
    report["status"] = "passed"
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("MEGAKERNEL_PASS", flush=True)


if __name__ == "__main__":
    main()
