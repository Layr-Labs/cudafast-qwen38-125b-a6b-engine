"""Actual short-tail helper/bridge controls with CUDA resolver/launch facades.
The host source slice tests local decline/error/completion routing, not the
preceding normalization/down/SiLU or GPU execution. No simulated arithmetic.
"""
from pathlib import Path
import re,subprocess,tempfile
r=Path(__file__).resolve().parents[1];s=(r/'ds4_cuda_qwen4exp.cu').read_text();m=(r/'ds4_cuda.cu').read_text()
def body(text,name):
 z=re.search(r'^.*\b'+name+r'\([^;]*?\)\s*\{',text,re.M);assert z,name
 a=z.start();e=text.index('{',a)+1;d=1
 while d:d+=(text[e]=='{')-(text[e]=='}');e+=1
 return text[a:e]
helper=body(s,'qwen4exp_hc_up_mix_short_try')
bridge=body(m,'cuda_q8_use_dp4a')+body(m,'ds4_cuda_qwen4exp_hc_up_warp_active')
f=body(s,'qwen4exp_hc_mixer_fused_cuda');a=f.index('        const int short_tail');b=f.index('    } else {',a);route=f[a:b]
source=r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <initializer_list>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int tier;};
struct ds4_gpu_qwen4exp_slab{const void*map;uint64_t map_size,offset,row_bytes;uint32_t type;};
constexpr uint32_t DS4_QWEN4EXP_TY_f32=0,DS4_QWEN4EXP_TY_q8_0=8;
static unsigned resolves,launches,old_calls,cases;static int resolve_error,launch_error,old_error;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
static const char*cuda_resolve_weight_ptr(const void*p,uint64_t o,uint64_t,int,const char*){resolves++;return resolve_error?nullptr:(const char*)((uintptr_t)p+o);}
static int cudaGetLastError(){return launch_error;}
static int cuda_ok(int e,const char*){return e==0;}
#define QWEN4EXP_LAUNCH_PDL(...) do{launches++;}while(0)
template<class...T>int ds4_gpu_matmul_q8_0_preq_rows_exact_tensor(T...){old_calls++;return !old_error;}
'''+bridge+helper+r'''
struct F{
 ds4_gpu_tensor mix{(void*)0x100000,20480,0},inj{(void*)0x200000,32,0},q{(void*)0x300000,81920,0},low{(void*)0x400000,2560,0},wide{(void*)0x500000,81920,0},hy{(void*)0x600000,81920,0};
 ds4_gpu_qwen4exp_slab up{(void*)0x1000000,3481600,0,340,8},iwdesc{(void*)0x2000000,40960*4,0,40960,0};
 const float*nw=(float*)0x3000000;const char*iw=(char*)0x2000000;
 uint64_t soff=20480;uint32_t rows=2,embd=2560,hc=4,lr=320;int staged=1;bool noin=false;
 int run(){
 auto*mixed=&mix;auto*inject=noin?nullptr:&inj;auto*normed_scratch=&q;auto*lowrank_scratch=&low;auto*wide_scratch=&wide;auto*hyper=&hy;auto*up_weight=&up;auto*inject_weight=noin?nullptr:&iwdesc;
 const float*normw=nw;const float*nscale=(float*)((char*)q.ptr+23040);uint64_t iw_row_bytes=iwdesc.row_bytes,s_off=soff;
 uint32_t n_embd=embd,n_hc=hc,n_lowrank=lr;uint64_t wide=(uint64_t)n_embd*n_hc;float weight_bias=1;int round_bf16=1;
'''+route+r'''
 return 1;
 }
};
void ck(bool b){cases++;if(!b){fprintf(stderr,"FAIL %u\n",cases);abort();}}
void reset(){resolves=launches=old_calls=0;resolve_error=launch_error=old_error=0;}
void eligible(F&a){reset();ck(a.run()==1&&launches==1&&old_calls==0);}
void decline(F&a){reset();ck(a.run()==1&&launches==0&&old_calls==1);old_error=1;old_calls=0;ck(a.run()==0&&launches==0&&old_calls==1);}
int main(){F a;eligible(a);a.rows=1;eligible(a);a.rows=2;a.noin=true;eligible(a);a.noin=false;a.iwdesc.type=8;eligible(a);a.up.offset=2;a.up.map_size+=2;eligible(a);
 for(const char*n:{"DS4_CUDA_NO_Q8_DP4A","DS4_QWEN4EXP_NO_ROW_TILE","DS4_Q8_NO_STREAM_LOADS","DS4_Q8_NO_HC_WARP_PAIR","DS4_QWEN4EXP_NO_HC_UP_MIX_SHORT","DS4_QWEN4EXP_NO_HC_DUAL"}){F b;setenv(n,"1",1);decline(b);unsetenv(n);}
 for(int k=0;k<18;k++){F b;switch(k){case 0:b.rows=0;break;case 1:b.rows=3;break;case 2:b.rows=48;break;case 3:b.embd=1280;break;case 4:b.hc=8;break;case 5:b.lr=640;break;case 6:b.staged=0;break;case 7:b.iwdesc.type=7;break;case 8:b.up.offset=1;b.up.map_size++;break;case 9:b.soff=20481;break;case 10:b.q.ptr=(void*)0x300004;break;case 11:b.inj.tier=1;break;case 12:b.q.bytes=100;break;case 13:b.iwdesc.row_bytes=UINT64_MAX;break;case 14:b.iwdesc.row_bytes=40959;break;case 15:b.mix.ptr=nullptr;break;case 16:b.inj.ptr=(void*)0x200002;break;case 17:b.wide.ptr=(char*)b.inj.ptr+4;break;}decline(b);}
 // Every mutable-output/new-read relationship plus skipped-wide aliases.
 for(int k=0;k<20;k++){F b;switch(k){case 0:b.mix.ptr=b.q.ptr;break;case 1:b.mix.ptr=b.low.ptr;break;case 2:b.mix.ptr=b.hy.ptr;break;case 3:b.mix.ptr=(void*)b.up.map;break;case 4:b.mix.ptr=(void*)b.nw;break;case 5:b.mix.ptr=(void*)b.iw;break;case 6:b.inj.ptr=b.q.ptr;break;case 7:b.inj.ptr=b.low.ptr;break;case 8:b.inj.ptr=b.hy.ptr;break;case 9:b.inj.ptr=(void*)b.up.map;break;case 10:b.inj.ptr=(void*)b.nw;break;case 11:b.inj.ptr=(void*)b.iw;break;case 12:b.inj.ptr=b.mix.ptr;break;case 13:b.wide.ptr=b.q.ptr;break;case 14:b.wide.ptr=b.low.ptr;break;case 15:b.wide.ptr=b.hy.ptr;break;case 16:b.wide.ptr=b.mix.ptr;break;case 17:b.wide.ptr=(void*)b.up.map;break;case 18:b.wide.ptr=(void*)b.nw;break;case 19:b.wide.ptr=(void*)b.iw;break;}decline(b);}
 {F b;b.mix.ptr=(void*)(UINTPTR_MAX-3);decline(b);}
 {F b;b.mix.ptr=(char*)b.hy.ptr+b.hy.bytes;eligible(b);}
 for(int k=0;k<4;k++){reset();F b;if(k==0)b.up.map=nullptr;if(k==1)b.up.map_size--;if(k==2)resolve_error=1;if(k==3)launch_error=1;ck(b.run()==0&&old_calls==0&&launches==(unsigned)(k==3));}
 printf("PASS short HC up/mix controls: %u checks; actual bridge/helper/local fallback, CUDA calls mocked\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='hc-upmix-controls-') as d:
 p=Path(d);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
