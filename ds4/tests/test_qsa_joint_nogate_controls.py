"""Actual old/new joint-wrapper validation and fallback with host facades.
Actual cached gate predicates run in fresh child processes for each diagnostic.
No GPU runtime or model is exercised by this test.
"""
from pathlib import Path
import re,subprocess,tempfile,os
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda_qwen4exp.cu').read_text();g=(repo/'ds4/ds4_qwen4exp_graph.inc').read_text()
def body(text,marker):
 a=text.index(marker);b=text.index('{',a);e=b+1;depth=1
 while depth:depth+=(text[e]=='{')-(text[e]=='}');e+=1
 return text[a:e]
wrappers=[]
for name in ['joint','joint_nogate']:
 api=body(s,'extern "C" int ds4_gpu_qwen4exp_qsa_prep_'+name+'_dpos_tensor(')
 api,n=re.subn(r'qwen4exp_qsa_prep_joint_kernel<(\d)><<<dim3\(qh\+kh,rows\),nth,nth\*4u,cuda_decode_stream\(\)>>>\((.*?)\);',lambda m:f'spy({m[1]}, qh+kh, rows, nth, nth*4u, cuda_decode_stream(), {m[2]});',api,flags=re.S);assert n==1;wrappers.append(api)
helpers='\n'.join(body(s,m) for m in ['static bool glm53_cuda_tensor_has(', 'static int qwen4exp_hc_ranges_disjoint(', 'static uint32_t qwen4exp_cuda_threads('])
predicates='\n'.join(body(g,m) for m in ['static bool qw_qsa_gate_q8(', 'static bool qw_qsa_gate_from_doubled('])
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;};
struct {uint64_t gdn_inner;} g_ds4_qwen4exp{6144};
static int called,part,qcopy,qnocopy,kv,error_code,failq,failkv;
static int cuda_decode_stream(){return 19;}static int cudaGetLastError(){return error_code;}static int cuda_ok(int e,const char*){return !e;}
template<class...T>static int ds4_gpu_qwen4exp_qsa_prep_q_fused_dpos_tensor(T...){qcopy++;return !failq;}
template<class...T>static int ds4_gpu_qwen4exp_qsa_prep_q_nogate_dpos_tensor(T...){qnocopy++;return !failq;}
template<class...T>static int ds4_gpu_qwen4exp_qsa_prep_kv_append_fused_dpos_tensor(T...){kv++;return !failkv;}
static void spy(int mode,unsigned grid,unsigned rows,unsigned nth,unsigned sh,int stream,const float*,const float*,const float*,const float*,const float*,const float*,float*,float*gate,float*,float*,float*,uint32_t r,uint32_t qh,uint32_t kh,uint32_t dim,uint32_t,uint32_t,uint32_t,float,float,float,const uint32_t*){
 called++;part=mode;assert(grid==qh+kh&&rows==r&&nth<=dim&&sh==nth*4&&stream==19);if(mode==4)assert(!gate);else assert(gate);
}
'''+helpers+'\n'+predicates+'\n'+'\n'.join(wrappers)+r'''
int main(int argc,char**argv){
 if(argc>1){bool enabled=!getenv("DS4_QWEN4EXP_NO_QSA_GATE_QUANT")&&!getenv("DS4_QWEN4EXP_NO_QSA_GATE_INPLACE");for(unsigned rows:{1u,2u,3u,8u})for(unsigned width:{512u,6144u})for(unsigned dim:{64u,256u})assert(qw_qsa_gate_from_doubled(rows,width,dim)==(enabled&&width==6144&&dim==256));puts("PASS cached consumer predicate");return 0;}
 unsigned count=0;ds4_gpu_tensor ob[5],ib[6],dp;ds4_gpu_tensor*out[5];const ds4_gpu_tensor*in[6];
 auto reset=[&](){for(unsigned i=0;i<5;i++){ob[i]={(void*)(uintptr_t)(0x10000000u+i*0x1000000u),1ull<<22};out[i]=&ob[i];}for(unsigned i=0;i<6;i++){ib[i]={(void*)(uintptr_t)(0x30000000u+i*0x1000000u),1ull<<22};in[i]=&ib[i];}dp={(void*)0x50000000,4};called=part=qcopy=qnocopy=kv=error_code=failq=failkv=0;unsetenv("DS4_QWEN4EXP_NO_QSA_PREP_JOINT");};
 auto call=[&](bool no,unsigned rows=2,unsigned dim=256,unsigned pos=0,unsigned cap=8,const ds4_gpu_tensor*p=nullptr){count++;return no?ds4_gpu_qwen4exp_qsa_prep_joint_nogate_dpos_tensor(out,in,rows,24,2,dim,64,pos,cap,1e-6f,1.f,1.f,p):ds4_gpu_qwen4exp_qsa_prep_joint_dpos_tensor(out,in,rows,24,2,dim,64,pos,cap,1e-6f,1.f,1.f,p);};
 reset();out[1]=(ds4_gpu_tensor*)(uintptr_t)1;assert(call(true)&&called==1);
 for(bool no:{false,true}){
  reset();assert(call(no)&&called==1&&part==(no?4:2));
  reset();out[4]=nullptr;assert(call(no)&&called==1);
  reset();out[1]=nullptr;assert(call(no)==no);assert(called==(no?1:0));
  reset();ob[1].bytes=0;ob[1].ptr=nullptr;assert(call(no)==no);
  for(unsigned idx=0;idx<5;idx++){if(no&&idx==1)continue;reset();ob[idx].bytes=1;assert(!call(no)&&!called&&!kv&&!qcopy&&!qnocopy);}
  for(unsigned idx=0;idx<6;idx++){reset();ib[idx].bytes=1;assert(!call(no)&&!called&&!kv);}
  reset();dp.bytes=0;assert(!call(no,2,256,0,8,&dp)&&!called);
  reset();assert(!call(no,2,256,7,8)&&!called);assert(call(no,2,256,7,8,&dp)&&called==1);
  reset();assert(!call(no,0)&&!called);assert(!call(no,65536)&&!called);
  for(unsigned reason=0;reason<5;reason++){
   reset();unsigned rows=2,dim=256;
   if(reason==0)setenv("DS4_QWEN4EXP_NO_QSA_PREP_JOINT","1",1);
   if(reason==1)rows=3;if(reason==2)dim=192;if(reason==3)ob[0].ptr=ib[0].ptr;if(reason==4)ob[2].ptr=ob[0].ptr;
   assert(call(no,rows,dim)&&!called&&kv==1&&qcopy==!no&&qnocopy==no);
   called=qcopy=qnocopy=kv=0;failq=1;assert(!call(no,rows,dim)&&!called&&!kv&&qcopy==!no&&qnocopy==no);
  }
  reset();ob[2].ptr=dp.ptr;assert(call(no,2,256,0,8,&dp)&&!called&&kv==1);
  reset();out[1]=&ob[0];assert(call(no)&&called==no); // Ignored gate never participates in new alias checks.
  reset();error_code=1;assert(!call(no)&&called==1&&!kv&&!qcopy&&!qnocopy);
  reset();setenv("DS4_QWEN4EXP_NO_QSA_PREP_JOINT","1",1);failkv=1;assert(!call(no)&&kv==1&&qcopy==!no&&qnocopy==no);
 }
 printf("PASS actual joint wrappers: %u capacity/position/null-gate/overlap/diagnostic/error cases\n",count);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-joint-controls-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
 for bits in range(4):
  env=os.environ.copy()
  for bit,name in enumerate(['DS4_QWEN4EXP_NO_QSA_GATE_QUANT','DS4_QWEN4EXP_NO_QSA_GATE_INPLACE']):
   env.pop(name,None)
   if bits&(1<<bit):env[name]='1'
  subprocess.run([str(p/'test'),'predicate'],env=env,check=True)
