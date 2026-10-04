from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--kernel-data',type=Path,required=True)
parser.add_argument('--architecture-data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args();options.output.mkdir(parents=True,exist_ok=True)
lab=tile=options.output
plt.rcParams.update({'font.size':10,'figure.facecolor':'#fbfbef','axes.facecolor':'#fbfbef','savefig.facecolor':'#fbfbef','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
data=json.loads(options.kernel_data.read_text());cases=data['assessment']
fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(cases));width=.19
variants=[('conventional-graph','Conventional CuTe','#778785'),('persistent-graph','Shared megakernel','#405b46'),('persistent-global-graph','Global intermediate','#b3a567'),('torch-cublas-graph','cuBLAS/PyTorch','#45454c')]
for i,(key,label,color) in enumerate(variants):
 values=[c['engines'][key]['stream_summary']['p50_ms']*1000 for c in cases]
 ax.bar(x+(i-1.5)*width,values,width,label=label,color=color)
ax.set_xticks(x,[' × '.join(str(c['shape'][k]) for k in ['rows','channels','hidden']) for c in cases])
ax.set_xlabel('Assessment shape: rows × channels × hidden width')
ax.set_ylabel('Median CUDA-event interval (µs)');ax.set_title('Residual MLP execution on one RTX 3090')
ax.legend(frameon=False,ncol=2,loc='upper left');ax.set_ylim(0,max(ax.get_ylim()[1],48))
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.text(.08,.025,'FP16 values / hidden state, FP32 accumulation and output. CUDA Graph replay for all variants.\n1,000 observations each; rounds share one process. Stream intervals may include dispatch idle gaps.',fontsize=8)
fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(lab/'gpu-kernel-assessment.svg');fig.savefig(lab/'gpu-kernel-assessment.png',dpi=160);plt.close(fig)
study=json.loads(options.architecture_data.read_text());all_points=study['architecture_sweep']['points']
# Identical predictions at larger scratch budgets do not imply a new performance point.
unique={}
for point in all_points:
 key=(point['latency_us'],point['energy_uj'],point['macs_per_cycle'])
 if key not in unique or point['area_proxy']<unique[key]['area_proxy']:unique[key]=point
points=list(unique.values())
fig,ax=plt.subplots(figsize=(9,5))
colors={8192:'#798980',16384:'#b2a467',32768:'#4d596b'}
for scratch,color in colors.items():
 selected=[p for p in points if p['scratch_bytes']==scratch]
 if not selected:continue
 ax.scatter([p['energy_uj'] for p in selected],[p['latency_us'] for p in selected],s=[p['macs_per_cycle']/2 for p in selected],alpha=.7,color=color,label=f'{scratch//1024} KiB minimum tested scratch')
pareto=[p for p in points if p['pareto']]
ax.scatter([p['energy_uj'] for p in pareto],[p['latency_us'] for p in pareto],s=90,facecolors='none',edgecolors='#292925',linewidths=1,label='Proxy Pareto points')
ax.set_xlabel('Estimated energy (µJ)');ax.set_ylabel('Estimated latency (µs)');ax.set_title('Architecture and tile sensitivity — analytical model')
ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
fig.text(.08,.025,'Original 17 × 65 × 79 MLP. Component rates/energy are explicit assumptions; area is a proxy.\nEstimates are not calibrated silicon measurements. Duplicate predictions use the smallest tested capacity.',fontsize=8)
fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(tile/'tile-architecture-study.svg');fig.savefig(tile/'tile-architecture-study.png',dpi=160)
print('Saved standalone charts.')
