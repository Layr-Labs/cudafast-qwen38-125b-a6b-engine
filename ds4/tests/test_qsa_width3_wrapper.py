"""Actual QSA triple API body with counted launch/runtime facades.
This proves host dispatch and error flow, not CUDA runtime/API execution.
"""
from pathlib import Path
import re,subprocess,tempfile
repo=Path(__file__).resolve().parents[2]
s=(repo/'ds4/ds4_cuda.cu').read_text();g=(repo/'ds4/ds4_qwen4exp_graph.inc').read_text()
def body(marker):
 a=s.index(marker);b=s.index('{',a);e=b+1;depth=1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
api=body('extern "C" int ds4_gpu_matmul_q8_0_preq_triple_rows_exact_tensor(')
api,n=re.subn(r'QWEN4EXP_LAUNCH_PDL\(\(qwen_q8_projection_triple_kernel<(\d)>\),\s*grid, 256, 0, cuda_decode_stream\(\),(.*?)\);',lambda m:f'spy({m[1]}, true, grid, 256, 0, cuda_decode_stream(), {m[2]});',api,flags=re.S);assert n==2
api,n=re.subn(r'qwen_q8_projection_triple_kernel<3><<<grid, 256, 0,\s*cuda_decode_stream\(\)>>>\((.*?)\);',lambda m:f'spy(3, false, grid, 256, 0, cuda_decode_stream(), {m[1]});',api,flags=re.S);assert n==1
a=g.index('if ((n_tokens<=2u || (n_tokens==3u &&',g.index('/* Projections.  q is DOUBLED'))+4;p=a;depth=1
while depth:depth+=(g[p]=='(')-(g[p]==')');p+=1
graph=g[a:p-1]
overlap=body('static inline bool qwen_gdn_projection_overlap(')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int tier;};
static int g_n_gpus=1,called,fallback,resolves,seen_rows;static bool seen_pdl,dp,fail_launch,fail_fallback;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
static const char*cuda_resolve_weight_ptr(const void*p,uint64_t o,uint64_t,int,const char*){resolves++;return (const char*)p+o;}
static int cuda_q8_use_dp4a(){return dp;}static int cuda_decode_stream(){return 23;}
static int cudaGetLastError(){return fail_launch?1:0;}static int cuda_ok(int e,const char*){return !e;}
static int cuda_matmul_q8_0_preq_rows_exact(ds4_gpu_tensor*,const char*,const int8_t*,const float*,uint64_t,uint64_t,uint32_t,uint64_t){fallback++;return !fail_fallback;}
static void spy(unsigned r,bool pdl,unsigned grid,unsigned threads,unsigned sm,int stream,float*,float*,float*,const unsigned char*,const unsigned char*,const unsigned char*,const int8_t*,const float*,uint64_t o0,uint64_t o1,uint64_t o2,uint32_t rows,uint64_t){
 called++;seen_rows=r;seen_pdl=pdl;assert(r==rows&&threads==256&&sm==0&&stream==23&&grid==(o0+3)/4+(o1+3)/4+(o2+3)/4);
}
'''+overlap+'\n'+api+r'''
int main(){unsigned count=0;
 const char*envs[]={"DS4_QWEN4EXP_NO_WIDE_VERIFY","DS4_QWEN4EXP_WIDE_VERIFY_R2","DS4_QWEN4EXP_NO_ROW_TILE","DS4_QWEN4EXP_PAIR_LANES_R2","DS4_QWEN4EXP_Q8_WIDE_BLOCKS","DS4_QWEN4EXP_NO_QSA_Q8_TRIPLE"};
 ds4_gpu_tensor q{(void*)0x20000000,4*256*36,0};ds4_gpu_tensor out[3];ds4_gpu_tensor*outs[3];const void*maps[3];uint64_t sizes[3],offs[3];
 for(unsigned bits=0;bits<64;bits++)for(unsigned rows:{1u,2u,3u,4u})for(unsigned dim:{1u,513u,12288u})for(uint64_t in_dim:{320ull,2560ull,8192ull})for(unsigned off:{0u,1u,2u})for(bool supported:{false,true}){
  for(unsigned j=0;j<6;j++)if(bits&(1u<<j))setenv(envs[j],"1",1);else unsetenv(envs[j]);
  uint64_t od[3]={dim,dim==12288?512:dim+2,dim==12288?512:dim+4};
  for(unsigned i=0;i<3;i++){out[i]={(void*)(uintptr_t)(0x30000000u+i*0x10000000u),4*od[i]*4,0};outs[i]=&out[i];maps[i]=(void*)(uintptr_t)(0x80000000ull+i*0x10000000ull);offs[i]=off;sizes[i]=off+od[i]*(in_dim/32)*34;}
  called=fallback=resolves=0;dp=supported;fail_launch=fail_fallback=false;
  int ok=ds4_gpu_matmul_q8_0_preq_triple_rows_exact_tensor(outs,maps,sizes,offs,in_dim,od,&q,0,rows*in_dim,rows);
  bool width=rows<=2||(rows==3&&!(bits&3));bool fused=width&&in_dim!=320&&supported&&!(bits&60)&&!(off&1);
  unsigned n_tokens=rows;bool caller=('''+graph+r''');assert(caller==(width&&!(bits&32)));
  assert(ok&&resolves==3);
  if(fused)assert(called==1&&fallback==0&&seen_rows==(int)rows&&seen_pdl==(rows<=2));else assert(called==0&&fallback==3);
  called=fallback=0;fail_launch=fused;fail_fallback=!fused;
  assert(!ds4_gpu_matmul_q8_0_preq_triple_rows_exact_tensor(outs,maps,sizes,offs,in_dim,od,&q,0,rows*in_dim,rows));
  if(fused)assert(called==1&&fallback==0);else assert(called==0&&fallback==1);
  count++;
 }
 printf("PASS actual QSA wrapper/graph guards: %u dispatch cases plus launch/fallback errors\n",count);
}
'''
with tempfile.TemporaryDirectory(prefix='qsa-r3-wrapper-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
