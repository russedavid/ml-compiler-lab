"""Small original Ampere warp-MMA projection with guarded boundary tiles.

A transparent single-stage design for comparison and generated-code inspection.
It favors simplicity over CUTLASS's pipelined, swizzled high-throughput schedule.
"""

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils


@cute.kernel
def tensorcore_kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    tiled_mma: cute.TiledMma,
    with_residual: cutlass.Constexpr,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    a = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((tile_m, 16), stride=(16, 1)),
        byte_alignment=16,
    )
    b = smem.allocate_tensor(
        cutlass.Float16,
        cute.make_layout((tile_n, 16), stride=(16, 1)),
        byte_alignment=16,
    )
    thread_mma = tiled_mma.get_slice(tx)
    pa = thread_mma.partition_A(a)
    pb = thread_mma.partition_B(b)
    ra = tiled_mma.make_fragment_A(pa)
    rb = tiled_mma.make_fragment_B(pb)
    tile = cute.local_tile(out, (tile_m, tile_n), (bm, bn))
    pc = thread_mma.partition_C(tile)
    acc = tiled_mma.make_fragment_C(pc)
    acc.fill(0)
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    for kt in range(cute.ceil_div(x.shape[1], 16)):
        for i in range(tx, tile_m * 16, threads):
            row = i // 16
            kk = i % 16
            value = cutlass.Float16(0)
            if bm * tile_m + row < x.shape[0] and kt * 16 + kk < x.shape[1]:
                value = x[bm * tile_m + row, kt * 16 + kk]
            a[row, kk] = value
        for i in range(tx, tile_n * 16, threads):
            col = i // 16
            kk = i % 16
            value = cutlass.Float16(0)
            if bn * tile_n + col < w.shape[1] and kt * 16 + kk < w.shape[0]:
                value = w[kt * 16 + kk, bn * tile_n + col]
            b[col, kk] = value
        cute.arch.sync_threads()
        cute.copy(copy_atom, pa, ra)
        cute.copy(copy_atom, pb, rb)
        cute.gemm(tiled_mma, acc, ra, rb, acc)
        cute.arch.sync_threads()
    coordinates = thread_mma.partition_C(cute.make_identity_tensor((tile_m, tile_n)))
    for i in range(cute.size(acc)):
        pos = coordinates[i]
        row = bm * tile_m + pos[0]
        col = bn * tile_n + pos[1]
        if row < out.shape[0] and col < out.shape[1]:
            value = acc[i] + bias[col]
            if cutlass.const_expr(with_residual):
                value = value + residual[row, col]
            if value < cutlass.Float32(0):
                value = cutlass.Float32(0)
            out[row, col] = value


@cute.jit
def tensorcore_projection(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
    with_residual: cutlass.Constexpr,
    stream: cuda.CUstream,
    tile_m: cutlass.Constexpr = 16,
    tile_n: cutlass.Constexpr = 8,
):
    if cutlass.const_expr(tile_m >= 32 and tile_n >= 16):
        atom_layout = cute.make_layout((2, 2, 1))
        threads = 128
    else:
        atom_layout = cute.make_layout((1, 1, 1))
        threads = 32
    mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
        atom_layout,
    )
    tensorcore_kernel(
        x, w, bias, residual, out, mma, with_residual, tile_m, tile_n, threads
    ).launch(
        grid=(
            cute.ceil_div(x.shape[0], tile_m),
            cute.ceil_div(out.shape[1], tile_n),
            1,
        ),
        block=(threads, 1, 1),
        stream=stream,
    )


@cute.kernel
def half_cast_kernel(x: cute.Tensor, y: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < cute.size(x):
        y[i] = x[i].to(cutlass.Float16)


@cute.jit
def half_cast(x: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream):
    half_cast_kernel(x, y).launch(
        grid=(cute.ceil_div(cute.size(x), 256), 1, 1), block=(256, 1, 1), stream=stream
    )
