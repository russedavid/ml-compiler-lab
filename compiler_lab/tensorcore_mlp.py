"""Two-projection persistent megakernel using Ampere warp tensor cores.

Each CTA owns sixteen independent rows and all hidden channels. The intermediate
is rounded to FP16 in shared memory; accumulators, bias and final output are FP32.
"""

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils


@cute.jit
def _gemm_rows(
    x: cute.Tensor,
    w: cute.Tensor,
    a: cute.Tensor,
    b_storage: cute.Tensor,
    row_base: cutlass.Int32,
    columns: cutlass.Constexpr,
    mma: cute.TiledMma,
):
    tx, _, _ = cute.arch.thread_idx()
    b = cute.local_tile(b_storage, (columns, 16), (0, 0))
    tm = mma.get_slice(tx)
    pa = tm.partition_A(a)
    pb = tm.partition_B(b)
    ra = mma.make_fragment_A(pa)
    rb = mma.make_fragment_B(pb)
    coords = tm.partition_C(cute.make_identity_tensor((16, columns)))
    acc = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
    acc.fill(0)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    for kt in range(cute.ceil_div(x.shape[1], 16)):
        for i in range(tx, 256, 128):
            rr = i // 16
            kk = i % 16
            value = cutlass.Float16(0)
            if row_base + rr < x.shape[0] and kt * 16 + kk < x.shape[1]:
                value = x[row_base + rr, kt * 16 + kk]
            a[rr, kk] = value
        for i in range(tx, columns * 16, 128):
            col = i // 16
            kk = i % 16
            value = cutlass.Float16(0)
            if kt * 16 + kk < w.shape[0] and col < w.shape[1]:
                value = w[kt * 16 + kk, col]
            b[col, kk] = value
        cute.arch.sync_threads()
        cute.copy(atom, pa, ra)
        cute.copy(atom, pb, rb)
        cute.gemm(mma, acc, ra, rb, acc)
        cute.arch.sync_threads()
    return acc, coords


@cute.kernel
def persistent_tensorcore_kernel(
    x: cute.Tensor,
    w1: cute.Tensor,
    b1: cute.Tensor,
    w2: cute.Tensor,
    b2: cute.Tensor,
    out: cute.Tensor,
    workers: cutlass.Int32,
    mma: cute.TiledMma,
    global_hidden: cute.Tensor,
    use_global: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    worker, _, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()
    a = allocator.allocate_tensor(
        cutlass.Float16, cute.make_layout((16, 16), stride=(16, 1)), byte_alignment=16
    )
    padded_hidden = cute.ceil_div(w1.shape[1], 32) * 32
    padded_output = cute.ceil_div(w2.shape[1], 32) * 32
    b = allocator.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((max(padded_hidden, padded_output), 16), stride=(16, 1)),
        byte_alignment=16,
    )
    hidden = allocator.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((16, padded_hidden), stride=(padded_hidden, 1)),
        byte_alignment=16,
    )
    task = worker
    while task < cute.ceil_div(x.shape[0], 16):
        base = task * 16
        first, positions = _gemm_rows(x, w1, a, b, base, padded_hidden, mma)
        for i in range(cute.size(first)):
            pos = positions[i]
            value = cutlass.Float32(0)
            if pos[1] < w1.shape[1]:
                value = first[i] + b1[pos[1]]
                if value < cutlass.Float32(0):
                    value = cutlass.Float32(0)
            if cutlass.const_expr(use_global):
                global_hidden[base + pos[0], pos[1]] = value.to(cutlass.Float16)
            else:
                hidden[pos[0], pos[1]] = value.to(cutlass.Float16)
        cute.arch.sync_threads()
        if cutlass.const_expr(use_global):
            second, coords = _gemm_rows(global_hidden, w2, a, b, base, padded_output, mma)
        else:
            second, coords = _gemm_rows(hidden, w2, a, b, 0, padded_output, mma)
        for i in range(cute.size(second)):
            pos = coords[i]
            row = base + pos[0]
            col = pos[1]
            if row < out.shape[0] and col < out.shape[1]:
                value = second[i] + b2[col] + x[row, col].to(cutlass.Float32)
                if value < cutlass.Float32(0):
                    value = cutlass.Float32(0)
                out[row, col] = value
        cute.arch.sync_threads()
        task = task + workers


@cute.jit
def persistent_tensorcore_mlp(
    x: cute.Tensor,
    w1: cute.Tensor,
    b1: cute.Tensor,
    w2: cute.Tensor,
    b2: cute.Tensor,
    out: cute.Tensor,
    workers: cutlass.Int32,
    stream: cuda.CUstream,
):
    mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
        cute.make_layout((1, 4, 1)),
    )
    persistent_tensorcore_kernel(x, w1, b1, w2, b2, out, workers, mma, out, False).launch(
        grid=(workers, 1, 1), block=(128, 1, 1), stream=stream
    )


@cute.jit
def global_tensorcore_mlp(x: cute.Tensor,w1: cute.Tensor,b1: cute.Tensor,
                          w2: cute.Tensor,b2: cute.Tensor,out: cute.Tensor,
                          global_hidden: cute.Tensor,workers: cutlass.Int32,stream: cuda.CUstream):
    mma=cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16,cutlass.Float32,(16,8,16)),cute.make_layout((1,4,1)))
    persistent_tensorcore_kernel(x,w1,b1,w2,b2,out,workers,mma,global_hidden,True).launch(grid=(workers,1,1),block=(128,1,1),stream=stream)
