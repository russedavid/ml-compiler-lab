# Evaluation and failure handling

The numerical reference is an independently executable PyTorch or NumPy expression. Input shapes, dtype boundaries and numerical tolerances are fixed before a run. A successful compile is followed by complete-output checks. Wrong shapes, nonfinite output, mismatches and changed-request failures reject the candidate before timing.

The original random-weight workloads measure numerical equivalence and execution behavior. They do not establish semantic perception accuracy or demographic fairness. The software accepts only documented graph patterns; the frontend rejects reductions across rows, unsupported operators, precision and weight shapes before choosing a row-owned schedule.

## Development and assessment

Kernel development uses the 33×32×64 and 257×64×128 MLP shapes. Assessment should include fresh combinations of rows/channels/hidden sizes and seeds. Once an assessment failure becomes a development case, use new assessment cases for a subsequent generalization claim. Report every unsupported case and every regression; no aggregate score hides an incorrect output.

Both host-observed completion and CUDA-event intervals are retained. CUDA events can include idle time between host dispatches. CUDA Graph replay is therefore a required control for each execution strategy. Graph capture must use the capture stream, and replay must overwrite a poisoned output and respond to changed inputs. An empty graph is an invalid measurement.

## Retained failures

- Passing compile-time booleans to a compiled CuTe call produced a workspace-argument mismatch. The executor now removes those constants from runtime arguments.
- The first tile-tuning script captured an empty CUDA Graph by launching on a different stream. Those timings are invalid. The corrected script uses the current capture stream and verifies replay output.
- The first tensor-core megakernel left partial-channel MMA extents unpadded and failed with nonfinite results. It now pads shared tiles, initializes padded hidden entries and guards weight/output access. The fixed kernel is checked under memory, race and synchronization tools.
- The first IREE VM bridge used the high-level argument packer for util.buffer references, which did not support that conversion. The bridge now invokes the VM function through typed variant lists.

These are concrete development failures. They do not become passing examples by relaxing numerical checks.

## Responsible optimization

A dtype change is a different numerical contract. The tensor-core projection uses FP16 inputs/weights and FP32 accumulation, bias, residual and output. It is compared with a cuBLAS control using the same output dtype. It is not described as a speedup of an unchanged FP32 model.

A downstream application must evaluate task accuracy on licensed, representative data before choosing precision or transformations. This repository provides numerical and execution tests; it does not provide a pretrained perception model or a representative human-performance/bias study. No private camera recordings are bundled or uploaded.

The default execution experiments are local. CPU CI runs on hosted infrastructure; GPU checks are explicitly invoked on a controlled machine. There is no public self-hosted runner accepting arbitrary pull-request code on a private workstation.

## AI-assisted development

AI assistance was used in implementation and review, including diagnosis of runtime-argument and capture-stream defects. Deterministic numerical tests and sanitizer runs are the acceptance checks. This release does not claim a measured improvement in developer productivity: there is no matched unassisted human baseline. The failure record is evidence of the development process, not a fabricated benchmark of an assistant.

## Evaluate your model and data

`compiler_lab.quality` accepts two callables and supplied data. Its CLI factory returns `(reference, candidate)` functions that accept one NumPy input and return an output array. A dataset NPZ contains `inputs`, and optionally `labels` and string `groups`. Classification labels require one-dimensional logits per example. The report retains every numerical failure, reference/candidate task accuracy when labels are available, and results per supplied slice.

```sh
.venv/bin/python -m compiler_lab.quality --factory your_model:make_pair \
  --data your_evaluation.npz --output runs/quality/report.json
```

Keep data provenance, license and slice selection with the dataset. Save evaluation outputs locally; do not publish private inputs or labels. Set aside assessment data before tuning the candidate.
