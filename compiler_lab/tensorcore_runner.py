from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from .tensorcore import tensorcore_projection, half_cast
from .tensorcore_mlp import persistent_tensorcore_mlp, global_tensorcore_mlp


class TensorCoreMLPExecutor:
    def __init__(self, tensors, strategy="persistent", workers=None):
        x, w1, b1, w2, b2 = (t.cuda().contiguous() for t in tensors)
        self.tensors = (x.half(), w1.half(), b1.float(), w2.half(), b2.float())
        x, w1, b1, w2, b2 = self.tensors
        if max(x.shape[1], w1.shape[1]) > 256:
            raise ValueError("channels/hidden exceed supported bound")
        self.output = torch.empty_like(x, dtype=torch.float32)
        self.hidden = torch.empty(x.shape[0], w1.shape[1], device="cuda")
        self.hidden_half = torch.empty_like(self.hidden, dtype=torch.float16)
        self.calls = []
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if strategy in ("persistent", "persistent-global"):
            workers = min(
                (x.shape[0] + 15) // 16,
                workers or torch.cuda.get_device_properties(0).multi_processor_count,
            )
            args = tuple(from_dlpack(t) for t in (*self.tensors, self.output))
            entry = persistent_tensorcore_mlp
            if strategy == "persistent-global":
                self.global_hidden = torch.empty(((x.shape[0] + 15) // 16 * 16, (w1.shape[1] + 31) // 32 * 32), device="cuda", dtype=torch.float16)
                args = (*args, from_dlpack(self.global_hidden))
                entry = global_tensorcore_mlp
            fn = cute.compile(
                entry,
                *args,
                workers,
                stream,
                options="--generate-line-info",
            )
            self.calls.append((fn, (*args, workers)))
        elif strategy == "conventional":
            first = tuple(from_dlpack(t) for t in (x, w1, b1, x, self.hidden))
            fn = cute.compile(
                tensorcore_projection,
                *first,
                False,
                stream,
                32,
                16,
                options="--generate-line-info",
            )
            self.calls.append((fn, first))
            cast = tuple(
                from_dlpack(t)
                for t in (self.hidden.flatten(), self.hidden_half.flatten())
            )
            self.calls.append((cute.compile(half_cast, *cast, stream), cast))
            second = tuple(
                from_dlpack(t) for t in (self.hidden_half, w2, b2, x, self.output)
            )
            fn = cute.compile(
                tensorcore_projection,
                *second,
                True,
                stream,
                32,
                16,
                options="--generate-line-info",
            )
            self.calls.append((fn, second))
        else:
            raise ValueError("unknown strategy")

    def __call__(self):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        for fn, args in self.calls:
            fn(*args, stream)
        return self.output


def tensorcore_mlp_reference(x, w1, b1, w2, b2):
    h = (torch.mm(x, w1, out_dtype=torch.float32) + b1).relu().half()
    return (torch.mm(h, w2, out_dtype=torch.float32) + b2 + x.float()).relu()
