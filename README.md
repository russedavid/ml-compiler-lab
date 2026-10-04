# ML Compiler Lab

Compare a neural-network graph executed as several GPU kernels with a persistent megakernel that keeps its intermediate in shared memory.

The project contains CuTe DSL projections and residual MLP kernels for Ampere GPUs, CUDA Graph controls, a verified PyTorch-to-MLIR scheduling frontend, and a C++ scheduling pass built into IREE. It also exports CNN, attention and recurrent workloads through the stock IREE compiler so generated IR and GPU code can be inspected.

The persistent tensor-core kernel gives each thread block sixteen independent rows and their complete two-layer graph. Both projections run in one launch. It uses block-local synchronization and a grid-stride task loop; progress does not depend on another block being resident.

## Install

The GPU path requires Linux, Python 3.12 and an Ampere-or-newer NVIDIA GPU. The verified target is an RTX 3090 (SM86), driver 580.105.08. The pinned environment uses CUDA-12-compatible wheels.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

CPU tests require PyTorch but do not launch CUDA kernels. GPU validation uses the installed CUDA Compute Sanitizer:

```sh
.venv/bin/python check_kernels.py
.venv/bin/python check_tensorcore.py
.venv/bin/python check_tensorcore_mlp.py
COMPUTE_SANITIZER=/path/to/compute-sanitizer bash scripts/validate_gpu.sh
```

## Run the experiments

Use a fresh output directory for every benchmark. Run on an otherwise idle GPU; compiler builds and other CPU activity should also be quiet for latency measurements.

```sh
# Original CNN: PyTorch eager/compile and IREE CPU/CUDA.
.venv/bin/python -m compiler_lab.baseline --output runs/cnn-001 --torch-compile

# FP32 scalar graph: separate launches, fusion, persistence, memory-reuse ablation and graphs.
.venv/bin/python -m compiler_lab.benchmark --output runs/f32-001

# FP16 tensor-core projections with FP32 accumulation/output and cuBLAS controls.
.venv/bin/python -m compiler_lab.tensorcore_benchmark --output runs/projection-001

# Complete two-projection tensor-core megakernel, conventional kernels and cuBLAS controls.
.venv/bin/python -m compiler_lab.megakernel_benchmark --output runs/megakernel-001
.venv/bin/python -m compiler_lab.megakernel_benchmark --output runs/assessment-001 --assessment

# Companion attention and three-step recurrent workloads.
.venv/bin/python -m compiler_lab.companion_checks --output runs/companions-001
```

Graph replay is tested with poisoned outputs and changed inputs. Shape tails, mixed-precision boundaries and nondefault streams are part of validation. Failed numerical results are not timed as successes.

## Build and use the C++ compiler pass

A C++17 compiler, CMake and Ninja are needed. The build script pins IREE and its LLVM submodule to the source revision used by the baseline wheel. It builds the CUDA target and the out-of-tree `lab` plugin. Allow substantial time and disk space for the source build.

```sh
BUILD_JOBS=8 bash scripts/build_iree.sh
.venv/bin/python scripts/test_compiler.py --iree-opt .cache/iree-build/tools/iree-opt
.venv/bin/python -m compiler_lab.compiled_run \
  --iree-opt .cache/iree-build/tools/iree-opt --output runs/compiled-001
```

The frontend verifies the exact FX graph, then emits scheduling IR. The C++ pass selects a bounded row-owned FP32 kernel or a conventional fallback. The tensor-core megakernel has a separate, explicit mixed-precision contract; selecting it does not silently change the FP32 model. [Compiler design](docs/compiler.md).

## Measurements and limits

The original workloads use synthetic inputs and random weights. They establish numerical equivalence and execution behavior, not perception accuracy or whole-LLM serving performance.

The CNN baseline measures NumPy host input through synchronized host output, including transfers and allocation. Kernel studies use device-resident input/output and retain CUDA-event intervals and host completion times. Events can include dispatch idle gaps; each strategy therefore has a CUDA Graph replay control. Repeated rounds in one process are preliminary observations, not independent generalization evidence.

The FP16 megakernel rounds input/weights and hidden state to FP16, with FP32 accumulation, bias and output. Comparisons preserve that contract. Results include shapes where fusion or persistence loses. There is no universal speedup claim.

[Recorded results](docs/results.md) · [Evaluation and retained failures](docs/evaluation.md) · [Kernel experiment design](docs/kernel-experiments.md)
