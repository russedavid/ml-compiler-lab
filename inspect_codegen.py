from pathlib import Path
from cuda.bindings import driver as cuda
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from compiler_lab.tensorcore import tensorcore_projection

x = torch.randn(17, 31, device="cuda", dtype=torch.float16)
w = torch.randn(31, 47, device="cuda", dtype=torch.float16)
b = torch.zeros(47, device="cuda")
r = torch.zeros(17, 47, device="cuda")
o = torch.empty_like(r)
args = tuple(from_dlpack(t) for t in (x, w, b, r, o))
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
fn = cute.compile(
    tensorcore_projection, *args, True, stream, 32, 16, options="--generate-line-info"
)
print(type(fn).__name__)
print({k: type(v).__name__ for k, v in fn.__dict__.items()})
print(
    [
        n
        for n in dir(fn)
        if any(x in n for x in ("dump", "export", "module", "artifact"))
    ]
)
Path("runs/codegen").mkdir(exist_ok=True)
Path("runs/codegen/projection.o").write_bytes(fn.dump_to_object("projection"))
print("EXPORTED_OBJECT")
