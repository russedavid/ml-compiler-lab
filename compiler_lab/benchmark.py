"""Device-resident graph comparison with CUDA Graph and data-reuse controls."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time
import torch
from .mlp import MLPShape, make_mlp, reference_mlp
from .runner import MLPExecutor
from .metrics import compare, summarize


def measure(fn, samples, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    device = []
    host = []
    # Events bracket the stream interval, which can include dispatch idle gaps.
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        end.synchronize()
        host.append((time.perf_counter_ns() - t) / 1e6)
        device.append(start.elapsed_time(end))
    return device, host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        parser.error("positive samples/rounds required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    report = {
        "measurement": "Device-resident input/output and weights; host samples include event overhead and completion; CUDA-event intervals can include host dispatch idle gaps.",
        "precision": "float32; TF32 disabled",
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "packages": {p: metadata.version(p) for p in ["torch", "nvidia-cutlass-dsl"]},
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "cases": [],
        "rounds": args.rounds,
        "samples_per_round": args.samples,
    }
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,memory.used",
            "--format=csv",
        ],
        text=True,
    )
    (args.output / "hardware.csv").write_text(hardware)
    shapes = [
        MLPShape(1, 7, 13),
        MLPShape(33, 32, 64),
        MLPShape(129, 31, 47),
        MLPShape(257, 64, 128),
        MLPShape(1024, 128, 256),
    ]
    for shape in shapes:
        print("CASE", shape, flush=True)
        tensors = make_mlp(shape)
        expected = reference_mlp(*tensors).numpy()
        eager_tensors = tuple(t.cuda() for t in tensors)
        engines = {}
        fixed_outputs = {}
        objects = []
        for strategy in ["separate", "fused", "persistent-global", "persistent"]:
            runner = MLPExecutor(tensors, strategy, workers=16)
            engines[strategy] = runner
            fixed_outputs[strategy] = runner.output
            objects.append(runner)

        def eager():
            return reference_mlp(*eager_tensors)

        engines["torch-eager"] = eager
        for name in list(engines):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    engines[name]()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_output = engines[name]()
            objects.extend([graph, graph_output])

            def replay(graph=graph, output=graph_output):
                graph.replay()
                return output

            engines[name + "-graph"] = replay
            fixed_outputs[name + "-graph"] = graph_output
        entry = {"shape": shape.__dict__, "engines": {}}
        for name, fn in engines.items():
            if name in fixed_outputs:
                fixed_outputs[name].fill_(float("nan"))
            check = compare(fn().cpu().numpy(), expected)
            if not check["passed"]:
                raise AssertionError((shape, name, check))
            entry["engines"][name] = {
                "correctness": check,
                "stream_ms": [],
                "host_ms": [],
            }
        with torch.inference_mode():
            names = list(engines)
            for round_no in range(args.rounds):
                order = names[round_no % len(names) :] + names[: round_no % len(names)]
                for name in order:
                    device, host = measure(engines[name], args.samples, 20)
                    result = entry["engines"][name]
                    result["stream_ms"].append(device)
                    result["host_ms"].append(host)
            # Repeat with changed values and poisoned intermediates after timing.
            for tensor in [tensors[0]]:
                tensor.neg_()
            changed = reference_mlp(*tensors).numpy()
            eager_tensors[0].copy_(tensors[0])
            for runner in objects:
                if isinstance(runner, MLPExecutor):
                    runner.tensors[0].copy_(tensors[0])
                    runner.hidden.fill_(float("nan"))
                    runner.output.fill_(float("nan"))
            for name, fn in engines.items():
                if name in fixed_outputs:
                    fixed_outputs[name].fill_(float("nan"))
                check = compare(fn().cpu().numpy(), changed)
                if not check["passed"]:
                    raise AssertionError(("changed input", shape, name, check))
                result = entry["engines"][name]
                result["changed_input"] = check
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
            {
                n: round(r["stream_summary"]["p50_ms"] * 1000, 2)
                for n, r in entry["engines"].items()
            },
            flush=True,
        )
    report["status"] = "passed"
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("BENCHMARK_PASS", flush=True)


if __name__ == "__main__":
    main()
