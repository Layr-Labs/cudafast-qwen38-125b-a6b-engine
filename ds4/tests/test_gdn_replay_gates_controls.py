"""Actual run-wrapper with counted launch/runtime facades; no GPU execution.
Covers scratch contract, early-source alias fallback, launch failures and order.
"""
from pathlib import Path
import re, subprocess, tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda_qwen4exp.cu').read_text()
def body(mark):
 a=s.index(mark);b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
# Guard the duplicated kernels against accidental arithmetic changes.
old=body('__global__ static void qwen4exp_gdn_conv_kernel(')
new=body('__global__ static void qwen4exp_gdn_conv_replay_gates_kernel(')
old=old[old.index('{')+1:old.rindex('}')].strip()
new=new[new.index('{')+1:new.index('    /* One publisher per head/token.')].strip()
assert old==new, 'serial convolution changed before publication tail'
old=body('__global__ static void qwen4exp_gdn_replay_kernel(')
new=body('__global__ static void qwen4exp_gdn_replay_gates_kernel(')
old=old[old.index('{')+1:old.rindex('}')]
new=new[new.index('{')+1:new.rindex('}')]
old=old.replace('    const float decay_coeff = n_tokens ? a_log[head] : 0.0f;\n    const float bias = n_tokens ? dt_bias[head] : 0.0f;\n','')
old=old.replace('g = expf(decay_coeff * qwen4exp_gdn_softplus(raw_alpha[gate] + bias));\n                beta = qwen4exp_gdn_sigmoid(raw_beta[gate]);','const float2 pair = gate_pairs[gate];\n                g = pair.x; beta = pair.y;')
assert old==new, 'scalar replay changed beyond current gate source'
api=body('static int qwen4exp_cuda_gdn_run(')
# Only launch syntax is translated; arguments and branch/error control stay real.
api,n=re.subn(r'(qwen4exp_\w+)(?:<[^>]+>)?<<<.*?>>>(\s*)\(',lambda m:'spy("'+m[1]+'", ',api,flags=re.S)
assert n==13,n
helper=body('static int qwen4exp_replay_gate_disjoint(')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <initializer_list>
#define QWEN4EXP_GDN_DIM 128u
#define QWEN4EXP_GDN_HISTORY 3u
#define QWEN4EXP_GDN_ADOPT_SLOTS 6u
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u
#define DS4_QWEN4EXP_GDN_HEADS_TILED 1u
#define QWEN4EXP_GDN_CONV_PARALLEL_MIN_TOKENS 64u
#define QWEN4EXP_GDN_OCTET_ROWS 2u
struct alignas(8) float2{float x,y;};
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int owner,device_id;};
struct ds4_gpu_qwen4exp_slab{const void*map;uint64_t map_size,offset;};
struct ds4_gpu_qwen4exp_gdn_replay{ds4_gpu_tensor*checkpoint,*tape;const ds4_gpu_tensor*control;ds4_gpu_tensor*gate_scratch;uint32_t defer_rows;};
struct dim3{dim3(unsigned,unsigned,unsigned){}};
using cudaStream_t=int;
static std::vector<std::string>calls;static int queries,fail_query,wrong_device,wrong_type,fail_launch;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t?t->device_id:0;}
static int glm53_cuda_mul_u64(uint64_t a,uint64_t b,uint64_t*out){if(b&&a>UINT64_MAX/b)return 0;*out=a*b;return 1;}
static int glm53_cuda_tensor_has(const ds4_gpu_tensor*t,uint64_t n,uint64_t z){return t&&t->ptr&&z&&n<=t->bytes/z;}
static const float*qwen4exp_gdn_weight_f32(const void*p,uint64_t,uint64_t o,uint64_t,int,const char*){return (const float*)((uintptr_t)p+o);}
static int cuda_decode_stream(){return 0;}static float*qwen4exp_conv_scratch(int,uint64_t){return nullptr;}
static int cuda_ok(int e,const char*){return !e;}static int cudaGetLastError(){return fail_launch&&(int)calls.size()==fail_launch;}
static int cudaMemcpyAsync(void*,const void*,uint64_t,int,int){assert(false);return 0;}
static const int cudaMemcpyDeviceToDevice=0,cudaMemoryTypeDevice=2,cudaMemoryTypeManaged=3;
struct cudaPointerAttributes{int type,device;};
static int cudaGetDevice(int*d){queries++;*d=0;return fail_query==1;}
static int cudaPointerGetAttributes(cudaPointerAttributes*a,const void*){queries++;a->type=wrong_type?0:2;a->device=wrong_device?1:0;return fail_query==2;}
template<class...T>static void spy(const char*n,T...){calls.push_back(n);}
'''+helper+'\n'+api+r'''
int main(){unsigned cases=0;const uint64_t cd=10240,vd=6144,state=48ull*128*128;
 ds4_gpu_tensor t[15],saved[15];
 uint64_t sizes[]={2*vd*4,3*cd*4,state*4,6*3*cd*4,6*state*4,2*cd*4,96*4,96*4,2*vd*4,2*vd+2*vd/32*4,4,state*4,2*(2048+vd+96)*4,4,768};
 for(unsigned i=0;i<15;i++)t[i]={(void*)(uintptr_t)(0x10000000ull+i*0x10000000ull),sizes[i],1,0};
 ds4_gpu_qwen4exp_slab w[4],savedw[4];for(unsigned i=0;i<4;i++)w[i]={(void*)(uintptr_t)(0x200000000ull+i*0x10000000ull),0x1000000,0};
 for(unsigned i=0;i<15;i++)saved[i]=t[i];for(unsigned i=0;i<4;i++)savedw[i]=w[i];
 ds4_gpu_qwen4exp_gdn_replay rp{&t[11],&t[12],&t[13],&t[14]};
 auto reset=[&](){for(unsigned i=0;i<15;i++)t[i]=saved[i];for(unsigned i=0;i<4;i++)w[i]=savedw[i];rp.gate_scratch=&t[14];calls.clear();queries=fail_query=wrong_device=wrong_type=fail_launch=0;unsetenv("DS4_QWEN4EXP_NO_GDN_REPLAY_GATES");};
 auto run=[&](bool quant=true,unsigned nk=16,unsigned nv=48){cases++;return qwen4exp_cuda_gdn_run(&t[0],&t[1],&t[2],&t[3],&t[4],1,&t[5],&t[6],&t[7],&t[8],&w[0],&w[1],&w[2],&w[3],nk,nv,1,2,1,1e-6f,1e-6f,quant?&t[9]:nullptr,0,2*vd,&t[10],"test",&rp);};
 auto expect=[&](bool gates,bool quant=true){assert(calls.size()==3);assert(calls[0]==(gates?"qwen4exp_gdn_conv_replay_gates_kernel":"qwen4exp_gdn_conv_kernel"));assert(calls[1]==(gates?"qwen4exp_gdn_replay_gates_kernel":"qwen4exp_gdn_replay_kernel"));assert(calls[2]==(quant?"qwen4exp_gdn_output_quant_kernel":"qwen4exp_gdn_output_kernel"));};
 for(bool q:{false,true}){reset();assert(run(q));expect(true,q);assert(queries==2);}
 reset();rp.gate_scratch=nullptr;assert(run());expect(false);assert(!queries);
 reset();setenv("DS4_QWEN4EXP_NO_GDN_REPLAY_GATES","1",1);assert(run());expect(false);assert(queries==2);
 reset();assert(run(true,8,48));expect(false);
 for(unsigned bad=0;bad<9;bad++){reset();switch(bad){case 0:t[14].ptr=nullptr;break;case 1:t[14].bytes=767;break;case 2:t[14].ptr=(char*)t[14].ptr+4;break;case 3:t[14].device_id=1;break;case 4:wrong_device=1;break;case 5:wrong_type=1;break;case 6:fail_query=1;break;case 7:fail_query=2;break;case 8:t[14].ptr=(void*)(UINTPTR_MAX-7);break;}assert(!run());assert(calls.empty());}
 for(unsigned target=0;target<14;target++){reset();t[14].ptr=(char*)t[target].ptr+((t[target].bytes>=16)?8:0);assert(!run());assert(calls.empty());}
 for(unsigned target=0;target<4;target++){reset();t[14].ptr=(void*)((uintptr_t)w[target].map+8);assert(!run());assert(calls.empty());}
 // Moving reads earlier must never change legacy aliases: all 12 combinations.
 for(unsigned src=0;src<4;src++)for(unsigned dst:{1u,3u,5u}){reset();void*p=(char*)t[dst].ptr+8;if(src<2)t[6+src].ptr=p;else w[src-1].map=p;assert(run());expect(false);}
 reset();t[14].ptr=(char*)t[6].ptr+t[6].bytes;assert(run());expect(true);
 for(unsigned stage=1;stage<=3;stage++){reset();fail_launch=stage;assert(!run());assert(calls.size()==stage);}
 // Old three-member aggregates get a null optional field.
 ds4_gpu_qwen4exp_gdn_replay old={&t[11],&t[12],&t[13]};assert(!old.gate_scratch);
 printf("PASS actual replay gate wrapper: %u controls, all scratch/source overlaps, fallback and ordered failure short-circuit\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-replay-gate-controls-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
