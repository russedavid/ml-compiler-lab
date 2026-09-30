"""Reusable device-resident execution; no private-machine details in source."""

from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from .kernels import projection, activation, residual_activation, persistent_mlp


class MLPExecutor:
    def __init__(self, tensors, strategy="persistent", workers=16):
        self.tensors = tuple(t.cuda().contiguous() for t in tensors)
        x, w1, b1, w2, b2 = self.tensors
        if max(x.shape[1], w1.shape[1]) > 256:
            raise ValueError("row-owned kernel dimensions exceed 256")
        if (
            w1.shape != (x.shape[1], b1.numel())
            or w2.shape != (b1.numel(), x.shape[1])
            or b2.numel() != x.shape[1]
        ):
            raise ValueError("incompatible MLP tensor shapes")
        if any(t.dtype != torch.float32 for t in self.tensors):
            raise ValueError("this executor requires float32")
        self.strategy = strategy
        self.workers = max(1, min(workers, x.shape[0]))
        self.hidden = torch.empty(x.shape[0], w1.shape[1], device="cuda")
        self.hidden_raw = torch.empty_like(self.hidden)
        self.output = torch.empty_like(x)
        self.raw = torch.empty_like(x)
        self.calls = []
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        def add(function, tensors, *static):
            args = tuple(from_dlpack(t) for t in tensors) + static
            compiled = cute.compile(
                function, *args, stream, options="--generate-line-info"
            )
            runtime_args = tuple(from_dlpack(t) for t in tensors) + tuple(
                v for v in static if type(v) is not bool
            )
            self.calls.append((compiled, runtime_args))

        if strategy in ("persistent", "persistent-global"):
            add(
                persistent_mlp,
                (*self.tensors, self.hidden, self.output),
                self.workers,
                strategy == "persistent-global",
            )
        elif strategy == "fused":
            add(projection, (x, w1, b1, self.hidden), True, False)
            # Residual differs from the input to this second projection, so use
            # a separate final epilogue until a distinct residual input is supported.
            add(projection, (self.hidden, w2, b2, self.raw), False, False)
            add(
                residual_activation,
                (self.raw.flatten(), x.flatten(), self.output.flatten()),
            )
        elif strategy == "separate":
            add(projection, (x, w1, b1, self.hidden_raw), False, False)
            add(activation, (self.hidden_raw.flatten(), self.hidden.flatten()))
            add(projection, (self.hidden, w2, b2, self.raw), False, False)
            add(
                residual_activation,
                (self.raw.flatten(), x.flatten(), self.output.flatten()),
            )
        else:
            raise ValueError(f"unknown strategy {strategy}")

    def __call__(self):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        for fn, args in self.calls:
            fn(*args, stream)
        return self.output
