import torch
from compiler_lab.mlp import MLPShape,make_mlp
from compiler_lab.tensorcore_runner import TensorCoreMLPExecutor,tensorcore_mlp_reference

for shape in [MLPShape(5,19,37),MLPShape(193,63,111),MLPShape(513,96,192)]:
    values=make_mlp(shape,2718)
    for strategy in ['persistent','persistent-global']:
        runner=TensorCoreMLPExecutor(values,strategy)
        torch.testing.assert_close(runner(),tensorcore_mlp_reference(*runner.tensors),rtol=3e-4,atol=3e-4)
        runner.tensors[0].neg_();runner.output.fill_(float('nan'))
        if hasattr(runner,'global_hidden'):runner.global_hidden.fill_(float('nan'))
        torch.testing.assert_close(runner(),tensorcore_mlp_reference(*runner.tensors),rtol=3e-4,atol=3e-4)
        print(shape,strategy,'PASS',flush=True)
