"""Original SM86-compatible projection kernels and a row-independent megakernel.

Each CTA owns a complete row graph, so progress never depends on another CTA
being scheduled. Persistent workers visit rows with a grid-stride loop.
"""

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils


@cute.kernel
def projection_kernel(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
    activate: cutlass.Constexpr,
    add_residual: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    row, tile, _ = cute.arch.block_idx()
    col = tile * 128 + tx
    if col < output.shape[1]:
        total = cutlass.Float32(0)
        for k in range(x.shape[1]):
            total = total + x[row, k] * weight[k, col]
        total = total + bias[col]
        if cutlass.const_expr(add_residual):
            total = total + x[row, col]
        if cutlass.const_expr(activate):
            if total < cutlass.Float32(0):
                total = cutlass.Float32(0)
        output[row, col] = total


@cute.jit
def projection(
    x: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
    activate: cutlass.Constexpr,
    add_residual: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    projection_kernel(x, weight, bias, output, activate, add_residual).launch(
        grid=(output.shape[0], cute.ceil_div(output.shape[1], 128), 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.kernel
def activation_kernel(x: cute.Tensor, y: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < cute.size(x):
        value = x[i]
        if value < cutlass.Float32(0):
            value = cutlass.Float32(0)
        y[i] = value


@cute.jit
def activation(x: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream):
    activation_kernel(x, y).launch(
        grid=(cute.ceil_div(cute.size(x), 256), 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def residual_activation_kernel(x: cute.Tensor, residual: cute.Tensor, y: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < cute.size(x):
        value = x[i] + residual[i]
        if value < cutlass.Float32(0):
            value = cutlass.Float32(0)
        y[i] = value


@cute.jit
def residual_activation(
    x: cute.Tensor, residual: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream
):
    residual_activation_kernel(x, residual, y).launch(
        grid=(cute.ceil_div(cute.size(x), 256), 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def megakernel(
    x: cute.Tensor,
    w1: cute.Tensor,
    b1: cute.Tensor,
    w2: cute.Tensor,
    b2: cute.Tensor,
    intermediate: cute.Tensor,
    output: cute.Tensor,
    workers: cutlass.Int32,
    global_intermediate: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    worker, _, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()
    local_x = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((x.shape[1],)), byte_alignment=16
    )
    hidden = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((w1.shape[1],)), byte_alignment=16
    )
    row = worker
    while row < x.shape[0]:
        if tx < x.shape[1]:
            local_x[tx] = x[row, tx]
        cute.arch.sync_threads()
        if tx < w1.shape[1]:
            total = cutlass.Float32(0)
            for k in range(x.shape[1]):
                total = total + local_x[k] * w1[k, tx]
            total = total + b1[tx]
            if total < cutlass.Float32(0):
                total = cutlass.Float32(0)
            if cutlass.const_expr(global_intermediate):
                intermediate[row, tx] = total
            else:
                hidden[tx] = total
        cute.arch.sync_threads()
        if tx < x.shape[1]:
            total = cutlass.Float32(0)
            for k in range(w1.shape[1]):
                if cutlass.const_expr(global_intermediate):
                    value = intermediate[row, k]
                else:
                    value = hidden[k]
                total = total + value * w2[k, tx]
            total = total + b2[tx] + local_x[tx]
            if total < cutlass.Float32(0):
                total = cutlass.Float32(0)
            output[row, tx] = total
        # All consumers finish before this worker overwrites its shared buffers.
        cute.arch.sync_threads()
        row = row + workers


@cute.jit
def persistent_mlp(
    x: cute.Tensor,
    w1: cute.Tensor,
    b1: cute.Tensor,
    w2: cute.Tensor,
    b2: cute.Tensor,
    intermediate: cute.Tensor,
    output: cute.Tensor,
    workers: cutlass.Int32,
    global_intermediate: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    megakernel(
        x, w1, b1, w2, b2, intermediate, output, workers, global_intermediate
    ).launch(grid=(workers, 1, 1), block=(256, 1, 1), stream=stream)
