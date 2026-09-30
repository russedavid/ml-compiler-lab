"""Export and validate attention and recurrent workloads through stock IREE."""
import argparse,json,subprocess,sys,traceback
from pathlib import Path
import torch,numpy as np
import iree.runtime as rt
import iree.turbine.aot as aot
from .models import AttentionBlock,FixedGRU
from .metrics import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(2718)
    cases=[('attention',AttentionBlock().eval(),(torch.randn(1,17,32),)),
           ('gru',FixedGRU().eval(),(torch.randn(1,3,16),torch.randn(1,16)))]
    results=[];compiler=str(Path(sys.executable).parent/'iree-compile')
    for name,module,inputs in cases:
        result={'workload':name}
        try:
            directory=args.output/name;directory.mkdir()
            exported=torch.export.export(module,inputs)
            (directory/'graph.txt').write_text(str(exported.graph_module.graph))
            aot.export(exported).save_mlir(directory/'model.mlir')
            command=[compiler,str(directory/'model.mlir'),'--iree-hal-target-backends=cuda','--iree-cuda-target=sm_86','--iree-opt-level=O3','-o',str(directory/'model.vmfb')]
            proc=subprocess.run(command,capture_output=True,text=True);(directory/'compile.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError('compilation failed; see compile.log')
            config=rt.Config('cuda');compiled=rt.load_vm_module(rt.VmModule.copy_buffer(config.vm_instance,(directory/'model.vmfb').read_bytes()),config)
            with torch.inference_mode():reference=module(*inputs).numpy()
            actual=compiled.main(*(v.numpy() for v in inputs)).to_host()
            check=compare(actual,reference)
            result.update(status='passed' if check['passed'] else 'incorrect',correctness=check)
        except Exception:
            result.update(status='unsupported/error',traceback=traceback.format_exc())
        results.append(result)
        (args.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,result['status'],flush=True)
    if not all(r['status']=='passed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
