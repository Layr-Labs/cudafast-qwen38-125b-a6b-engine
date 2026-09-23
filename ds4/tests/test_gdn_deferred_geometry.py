#!/usr/bin/env python3
"""Synthetic geometry, loop-split and cache-hint experiments for folded GDN.

Modes: 0 legacy eager; 2 exact fixed128 baseline; 3/4 candidate256/512.
Mode1 is candidate64 for geometry alone, candidate128 with either optimization.
Use --source with the cd66b523 source when testing after production changes.
"""
import argparse
from pathlib import Path
import re
import subprocess
p=argparse.ArgumentParser()
p.add_argument('--nvcc',required=True)
p.add_argument('--source',type=Path)
p.add_argument('--arch',default='sm_89')
p.add_argument('--benchmark',action='store_true')
p.add_argument('--compile-only',action='store_true')
p.add_argument('--split-loop',action='store_true')
p.add_argument('--stream-state',action='store_true')
p.add_argument('--ablation',action='store_true')
a=p.parse_args()
if a.ablation:a.split_loop=a.stream_state=True
r=Path(__file__).resolve().parents[2]
out=r/('.build/gdn-ablation' if a.ablation else '.build/gdn-stream-split' if a.stream_state and a.split_loop else '.build/gdn-stream' if a.stream_state else '.build/gdn-loop-split' if a.split_loop else '.build/gdn-geometry')/a.arch
out.mkdir(parents=True,exist_ok=True)
s=(a.source or r/'ds4/ds4_cuda_qwen4exp.cu').read_text()
def body(name):
 pos=s.index(name+'(');start=s.rfind('\n',0,pos)+1;end=s.index('\n}',pos)+2
 return s[start:end]
# Only grid geometry changes. Each warp still owns exactly one value row.
k=body('qwen4exp_gdn_deferred_kernel').replace('qwen4exp_gdn_deferred_kernel','geometry_kernel')
assert k.count('blockIdx.y * 4u')==1
k=k.replace('blockIdx.y * 4u','blockIdx.y * (blockDim.x >> 5u)')
if a.split_loop:
    start=k.index('    for (uint32_t step = 0; step < prefix + n_tokens; step++) {')
    begin=k.index('{',start)+1
    end=begin;depth=1
    while depth:
        depth+=(k[end]=='{')-(k[end]=='}');end+=1
    loop=k[begin:end-1]
    setup='\n        const bool replay = step < prefix;\n        const uint32_t token = replay ? 0u : step - prefix;'
    assert loop.count(setup)==1
    replay=loop.replace(setup,'\n        const bool replay = true;\n        const uint32_t token = 0u;')
    current=loop.replace(setup,'\n        const bool replay = false;\n        const uint32_t step = prefix + token;')
    k=k[:start]+'    for (uint32_t step = 0; step < prefix; step++) {'+replay+'}\n    if (n_tokens != 0u) {\n#pragma unroll\n        for (uint32_t token = 0; token < 2u; token++) {'+current+'}\n    }'+k[end:]

if a.stream_state:
    k=k.replace('float4 h = *(const float4 *)(checkpoint + state_off);','float4 h = __ldcs((const float4 *)(checkpoint + state_off));')
    k=k.replace('*(float4 *)(checkpoint + state_off) = h;','__stcs((float4 *)(checkpoint + state_off), h);')
if a.ablation:k=k.replace('blockIdx.y * (blockDim.x >> 5u)','blockIdx.y * 4u')
header='#include <cuda_runtime.h>\n#include <cstdint>\n#include <cmath>\n#define QWEN4EXP_GDN_DIM 128u\n#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u\n'
base=body('qwen4exp_gdn_replay_gates_kernel')
materialize=base.replace('qwen4exp_gdn_replay_gates_kernel','materialize_kernel').replace('if (prefix > DS4_QWEN4EXP_GDN_REPLAY_ROWS) return;','if (prefix > 10u || n_tokens != 0u) return;')
header+='\n'.join(body(n) for n in ['warp_sum_all_f32','dot4_f32','qwen4exp_gdn_sigmoid','qwen4exp_gdn_softplus'])
header+='\n'+base+'\n'+materialize+'\ntemplate<bool Precomputed>\n'+k
header+='\ntemplate<bool Precomputed>\n'+body('qwen4exp_gdn_deferred_kernel').replace('qwen4exp_gdn_deferred_kernel','fixed_kernel')
if a.ablation:
    split=k.replace('geometry_kernel','split_kernel').replace('__ldcs((const float4 *)(checkpoint + state_off))','*(const float4 *)(checkpoint + state_off)').replace('__stcs((float4 *)(checkpoint + state_off), h);','*(float4 *)(checkpoint + state_off) = h;')
    cache=body('qwen4exp_gdn_deferred_kernel').replace('qwen4exp_gdn_deferred_kernel','cache_kernel').replace('*(const float4 *)(checkpoint + state_off)','__ldcs((const float4 *)(checkpoint + state_off))').replace('*(float4 *)(checkpoint + state_off) = h;','__stcs((float4 *)(checkpoint + state_off), h);')
    header+='\ntemplate<bool Precomputed>\n'+split+'\ntemplate<bool Precomputed>\n'+cache
header+='\n#define ABLATION '+str(int(a.ablation))+'\n'
header+='\n#define CANDIDATE_AT_128 '+str(int(a.split_loop or a.stream_state))+'\n'
(out/'kernels.cuh').write_text(header)
c=(r/'ds4/tests/test_gdn_deferred_state.cu').read_text()
c=c[:c.index('#ifdef GDN_DEFERRED_PRODUCTION\n__global__ void compute_pairs')]
start=c.index('template<unsigned Flush, bool Hybrid, bool FoldRow0>')
end=c.index('template<unsigned Flush, bool Hybrid=false',start)
c=c[:start]+"""template<unsigned Flush, bool Hybrid, bool FoldRow0, unsigned Threads=128>
void launch(float*out,float*state,float*cp,float*tape,float*qkv,float*gates,
            unsigned layout,uint32_t*control,cudaStream_t stream) {
    if constexpr (Threads==128) {
        fixed_kernel<true><<<dim3(NV,32),128,0,stream>>>(
            out,state,cp,tape,qkv,nullptr,nullptr,nullptr,nullptr,(float2*)gates,
            NK,NV,2,layout,control,0);
    } else {
#if ABLATION
        if constexpr (Threads==64) {
            split_kernel<true><<<dim3(NV,32),128,0,stream>>>(out,state,cp,tape,qkv,nullptr,nullptr,nullptr,nullptr,(float2*)gates,NK,NV,2,layout,control,0);
            return;
        }
        if constexpr (Threads==256) {
            cache_kernel<true><<<dim3(NV,32),128,0,stream>>>(out,state,cp,tape,qkv,nullptr,nullptr,nullptr,nullptr,(float2*)gates,NK,NV,2,layout,control,0);
            return;
        }
#endif
        constexpr unsigned T = ABLATION ? 128 : CANDIDATE_AT_128 && Threads==64 ? 128 : Threads;
        geometry_kernel<true><<<dim3(NV,128/(T/32)),T,0,stream>>>(
            out,state,cp,tape,qkv,nullptr,nullptr,nullptr,nullptr,(float2*)gates,
            NK,NV,2,layout,control,0);
    }
}

"""+c[end:]
c=c.replace('bool FoldRow0=false>','bool FoldRow0=false, unsigned Threads=128>')
c=c.replace('launch<Flush,Hybrid,FoldRow0>','launch<Flush,Hybrid,FoldRow0,Threads>')
c=c.replace('FoldRow0=%u: %u forwards','FoldRow0=%u Variant=%u: %u forwards').replace('unsigned(FoldRow0),checked','unsigned(FoldRow0),Threads,checked')
c=c.replace('control[8]','control[5]').replace('graph[8]','graph[5]').replace('ms[8]','ms[5]').replace('j<8','j<5').replace('(j+sample)%8','(j+sample)%5').replace('mode<8','mode<5')
args='a,b,log,qkv,gates,out,control[{i}],accepted,stream'
arms=['capture<0>('+args.format(i=0)+')']
arms += ['capture<2,false,1,true,'+str(t)+'>('+args.format(i=i+1)+')' for i,t in enumerate([64,128,256,512])]
c=re.sub(r'cudaGraphExec_t graph\[5\]=\{[^;]+;', 'cudaGraphExec_t graph[5]={'+','.join(arms)+'};',c)
c+='int main(int argc,char**argv){if(argc==2&&!strcmp(argv[1],"--benchmark"))benchmark();else{'+''.join('run<2,false,1,true,'+str(t)+'>();' for t in [64,128,256,512])+'}}\n'
(out/'test.cu').write_text(c)
cmd=[a.nvcc,'-O3','-std=c++17','-arch='+a.arch,'-ftz=false','-prec-div=true','-prec-sqrt=true','-Xptxas=-v','-I'+str(out),str(out/'test.cu')]
binary=out/('test.o' if a.compile_only else 'test')
if a.compile_only:cmd+=['-c']
with (out/'compile.log').open('w') as f:subprocess.run(cmd+['-o',str(binary)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Compiled',binary,flush=True)
if not a.compile_only:subprocess.run([str(binary)]+(['--benchmark'] if a.benchmark else []),check=True)
