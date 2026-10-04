# Kernel and megakernel design

Three original implementations expose different costs:

- A float32 SIMT projection and row-owned two-layer residual MLP.
- An Ampere tensor-core projection with guarded tiles and fused bias/residual/ReLU.
- A tensor-core megakernel that executes both MLP projections for sixteen independent rows per CTA, retaining the FP16 hidden state in shared memory.

The SIMT experiment compares separate operators, partial fusion, a persistent worker loop and a global-memory intermediate control. The tensor-core experiment preserves FP16 input/weight/hidden boundaries and FP32 accumulation/bias/output, with conventional CuTe and cuBLAS controls. Every strategy is also captured in a CUDA Graph.

## Layout and ownership

The tensor-core projection uses warp-level m16n8k16 MMA. Tile variants change the CTA footprint and warp distribution. Inputs at a tile boundary are zero-filled; final stores check the true dimensions.

The megakernel owns a complete row tile and every hidden channel needed by its second projection. Each CTA has its own operand staging and hidden-state buffers. Channel extents are padded to match the warp-MMA partition. Padded hidden entries are initialized, and weights/final outputs are guarded. An earlier unpadded variant failed a partial-channel case; the corrected layout is checked numerically and under sanitizers.

The worker loop visits more row tiles with a grid stride. It uses block barriers around shared producer/consumer phases and before overwriting buffers. It has no global spinning barrier, dependency on an unscheduled CTA or cross-GPU communication. The supported channels and hidden widths are bounded at 256.

## Capture and requests

Compiled CuTe calls receive the current CUDA stream explicitly. Compile-time constants are absent from runtime argument lists. Graph capture and replay use the correct stream; replay is checked by poisoning outputs and changing inputs. A captured graph that performs no computation is rejected.

Executors retain buffers for measurement and are intended for serialized requests. Use a separate executor for concurrent streams; they are not a multi-request serving system. Shape changes require a new specialization. First compilation is separate from warm timing.

## Interpreting performance

Graph replay controls for Python/launch overhead. The shared versus global intermediate controls distinguish persistence from memory reuse. Measurements retain cases where the persistent path loses. The initial C++ scheduling policy chooses the bounded FP32 route only for small development-supported shapes and otherwise uses a conventional fallback.

The single-stage tensor-core projection is deliberately inspectable: it is not CUTLASS's full asynchronous, swizzled GEMM pipeline and does not claim peak library throughput. Its generated PTX/CUBIN, register counts and shared allocation are available for diagnosis. The complete megakernel demonstrates a different scheduling/data-reuse tradeoff rather than a universal replacement for vendor kernels.

## References

- [NVIDIA CuTe DSL quick start and APIs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html).
- [CUTLASS Ampere examples](https://github.com/NVIDIA/cutlass/tree/v4.8.0/examples/python/CuTeDSL/cute/ampere).
- [Mirage Persistent Kernel](https://github.com/mirage-project/mirage/tree/mpk), prior art for graph-level persistent scheduling. This implementation does not reuse its multi-GPU engine or assume its newer-architecture kernels run on SM86.
