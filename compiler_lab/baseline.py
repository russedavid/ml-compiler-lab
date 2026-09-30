"""Export, compile and validate original CNN blocks on CPU and NVIDIA GPU.

The timing boundary is NumPy host input -> synchronous NumPy host output.
It includes transfer, allocation, dispatch and synchronization, not only kernels.
"""

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import iree.runtime as rt
import iree.turbine.aot as aot
from compiler_lab.models import make_case
from compiler_lab.metrics import compare, summarize

CASES = [("aligned", 16, 64, 64), ("tails", 7, 31, 47), ("larger", 32, 128, 128)]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(args, output):
    start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    output.write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"{args[0]} exited {proc.returncode}; see {output}")
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.rounds <= 0 or args.warmup < 0:
        parser.error("samples/rounds must be positive and warmup nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    compiler = str(Path(sys.executable).parent / "iree-compile")
    command([compiler, "--version"], args.output / "compiler-version.txt")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm",
            "--format=csv",
        ],
        text=True,
    )
    (args.output / "hardware-before.csv").write_text(gpu)
    packages = {
        p: metadata.version(p)
        for p in [
            "torch",
            "iree-turbine",
            "iree-base-compiler",
            "iree-base-runtime",
            "nvidia-cutlass-dsl",
            "numpy",
        ]
    }
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    report = {
        "status": "running",
        "packages": packages,
        "source_sha256": source_hashes,
        "gpu_arch": "sm_" + "".join(map(str, torch.cuda.get_device_capability())),
        "precision": "float32; PyTorch TF32 disabled",
        "seed": 1729,
        "tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "timing": {
            "boundary": "host NumPy input -> synchronized host NumPy output; weights loaded",
            "rounds": args.rounds,
            "samples_per_round": args.samples,
            "warmup": args.warmup,
            "limits": "Rounds share one process; preliminary baseline, not independent performance validation. No p99 claim.",
        },
        "semantic_accuracy": "Synthetic inputs and random weights; numerical equivalence only.",
        "cases": [],
    }
    write(args.output / "report.json", report)
    for name, channels, height, width in CASES:
        print("CASE", name, flush=True)
        case_dir = args.output / name
        case_dir.mkdir()
        model, x = make_case(channels, height, width)
        x_np = x.numpy()
        with torch.inference_mode():
            ref = model(x).numpy()
        np.save(case_dir / "input.npy", x_np)
        np.save(case_dir / "expected.npy", ref)
        np.savez(
            case_dir / "weights.npz",
            **{k: v.detach().numpy() for k, v in model.state_dict().items()},
        )
        entry = {"name": name, "shape": list(x.shape), "engines": {}, "comparisons": {}}
        report["cases"].append(entry)
        exported = torch.export.export(model, (x,))
        (case_dir / "graph.txt").write_text(str(exported.graph_module.graph))
        start = time.perf_counter()
        aot.export(exported).save_mlir(case_dir / "model.mlir")
        entry["export_seconds"] = time.perf_counter() - start
        functions = {}
        keepalive = []
        for backend, driver in [("llvm-cpu", "local-task"), ("cuda", "cuda")]:
            try:
                target_dir = case_dir / backend
                target_dir.mkdir()
                cmd = [
                    compiler,
                    str(case_dir / "model.mlir"),
                    f"--iree-hal-target-backends={backend}",
                    "--iree-opt-level=O3",
                    f"--iree-hal-dump-executable-files-to={target_dir / 'codegen'}",
                    "-o",
                    str(target_dir / "model.vmfb"),
                ]
                cmd += (
                    [f"--iree-cuda-target={report['gpu_arch']}"]
                    if backend == "cuda"
                    else ["--iree-llvmcpu-target-cpu=host"]
                )
                write(target_dir / "command.json", cmd)
                elapsed = command(cmd, target_dir / "compile.log")
                config = rt.Config(driver)
                module = rt.load_vm_module(
                    rt.VmModule.copy_buffer(
                        config.vm_instance, (target_dir / "model.vmfb").read_bytes()
                    ),
                    config,
                )
                keepalive.extend([config, module])
                fn = lambda data, module=module: module.main(data).to_host()
                start = time.perf_counter()
                actual = fn(x_np)
                first = time.perf_counter() - start
                check = compare(actual, ref)
                np.save(target_dir / "actual.npy", actual)
                result = {
                    "status": "passed" if check["passed"] else "incorrect",
                    "correctness": check,
                    "compile_seconds": elapsed,
                    "first_invocation_seconds": first,
                    "artifact_bytes": (target_dir / "model.vmfb").stat().st_size,
                }
                entry["engines"]["iree-" + backend] = result
                if check["passed"] and backend == "cuda":
                    functions["iree-cuda"] = fn
            except Exception:
                entry["engines"]["iree-" + backend] = {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                }
                print(traceback.format_exc(), flush=True)
        cuda_model = copy.deepcopy(model).cuda()
        with torch.inference_mode():

            def eager(data, model=cuda_model):
                return model(torch.from_numpy(data).cuda()).cpu().numpy()

            funcs = {"torch-eager": eager}
            if args.torch_compile:
                compiled = torch.compile(cuda_model, fullgraph=True)
                funcs["torch-compile"] = lambda data: (
                    compiled(torch.from_numpy(data).cuda()).cpu().numpy()
                )
            for engine, fn in funcs.items():
                try:
                    start = time.perf_counter()
                    actual = fn(x_np)
                    first = time.perf_counter() - start
                    check = compare(actual, ref)
                    entry["engines"][engine] = {
                        "status": "passed" if check["passed"] else "incorrect",
                        "correctness": check,
                        "first_invocation_seconds": first,
                        "first_invocation_includes_lazy_compilation": engine
                        == "torch-compile",
                    }
                    if check["passed"]:
                        functions[engine] = fn
                except Exception:
                    entry["engines"][engine] = {
                        "status": "error",
                        "traceback": traceback.format_exc(),
                    }
            # Check different inputs as well as the original seeded request.
            variants = [np.zeros_like(x_np), -x_np, np.ascontiguousarray(x_np * 2)]
            for engine, fn in list(functions.items()):
                checks = []
                for variant in variants:
                    expected = model(torch.from_numpy(variant)).numpy()
                    checks.append(compare(fn(variant), expected))
                entry["engines"][engine]["additional_inputs"] = checks
                if not all(c["passed"] for c in checks):
                    entry["engines"][engine]["status"] = "incorrect"
                    del functions[engine]
            for fn in functions.values():
                for _ in range(args.warmup):
                    fn(x_np)
            names = list(functions)
            samples = {n: [] for n in names}
            for round_no in range(args.rounds):
                order = (
                    names[round_no % len(names) :] + names[: round_no % len(names)]
                    if names
                    else []
                )
                for engine in order:
                    fn = functions[engine]
                    times = []
                    for _ in range(args.samples):
                        start = time.perf_counter_ns()
                        fn(x_np)
                        times.append((time.perf_counter_ns() - start) / 1e6)
                    samples[engine].append(times)
            for engine in names:
                result = entry["engines"][engine]
                result["rounds"] = [summarize(r) for r in samples[engine]]
                result["aggregate"] = summarize([v for r in samples[engine] for v in r])
                result["post_timing_correctness"] = compare(
                    functions[engine](x_np), ref
                )
                if not result["post_timing_correctness"]["passed"]:
                    result["status"] = "incorrect"
                write(case_dir / f"{engine}-samples.json", samples[engine])
        write(args.output / "report.json", report)
        print(
            json.dumps(
                {
                    k: {
                        "status": v["status"],
                        "p50_ms": v.get("aggregate", {}).get("p50_ms"),
                    }
                    for k, v in entry["engines"].items()
                }
            ),
            flush=True,
        )
        functions.clear()
        keepalive.clear()
        del cuda_model
        torch.cuda.empty_cache()
    report["status"] = (
        "passed"
        if all(
            e["status"] == "passed"
            for c in report["cases"]
            for e in c["engines"].values()
        )
        else "failures recorded"
    )
    write(args.output / "report.json", report)
    (args.output / "hardware-after.csv").write_text(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,power.draw,clocks.sm",
                "--format=csv",
            ],
            text=True,
        )
    )
    print(report["status"], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
