# Graph scheduling and generated GPU code

The Python frontend exports an exact residual MLP and checks its FX connectivity, operators, precision, weight shapes and output. It rejects cross-row reductions and unsupported forms. Only then does it emit `lab.graph` scheduling IR with the verified row-local contract and source graph hash.

The original C++ pass is built as a statically registered IREE compiler plugin. It lowers `lab.graph` into `lab.schedule`, selecting a bounded persistent route or a conventional fallback, with worker count, shared workspace and synchronization metadata. The initial policy is conservative and based on development cases; it is not a universal autotuner.

```sh
BUILD_JOBS=8 bash scripts/build_iree.sh
.cache/iree-build/tools/iree-opt --iree-plugin=lab \
  --lab-schedule-row-mlp compiler/tests/small.mlir -o runs/schedule.mlir
```

The pinned source build enables the CUDA backend and the lab plugin. CPU baseline experiments use the separately pinned IREE wheel. Compiler source, LLVM submodule and build settings are recorded by the build script. Generated artifacts and source checkouts stay under ignored directories.

The lab dialect represents a deliberately narrow graph contract, not arbitrary PyTorch programs. The CuTe kernels define the GPU layout, memory access, tensor-core operations and stage schedule. The CNN baseline uses the stock IREE compiler and retains its generated LLVM/PTX.
