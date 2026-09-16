"""Execute actual API body with counted CUDA/runtime facades, not GPU dispatch.
Only launch syntax is translated. Pointer resolution, ordinary/PDL selection,
validation, fallback, and error decisions are the production source statements.
"""
from pathlib import Path
import re,subprocess,tempfile
s=(Path(__file__).resolve().parents[1]/'ds4_cuda.cu').read_text()
def body(marker):
 a=s.index(marker);b=s.index('{',a);e=b+1;depth=1
 while depth:depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
api=body('extern "C" int ds4_gpu_qwen4exp_gdn_projections_exact_tensor(')
# Rewrite each launch only, retaining the actual arguments and runtime branches.
api,n=re.subn(r'QWEN4EXP_LAUNCH_PDL\(\(qwen_gdn_projection_kernel<(\d)(,true)?>\),\s*grid, 256, (gdn_panel|0), cuda_decode_stream\(\), a\);',lambda m:f'spy({m[1]}, {"true" if m[2] else "false"}, true, grid, 256, {m[3]}, cuda_decode_stream(), a);',api)
assert n==4
api,n=re.subn(r'qwen_gdn_projection_kernel<3(,true)?><<<grid, 256, (gdn_panel|0),\s*cuda_decode_stream\(\)>>>\(a\);',lambda m:f'spy(3, {"true" if m[1] else "false"}, false, grid, 256, {m[2]}, cuda_decode_stream(), a);',api)
assert n==2
args=body('struct qwen_gdn_projection_args')+';'
overlap=body('static inline bool qwen_gdn_projection_overlap(')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <initializer_list>
struct ds4_gpu_tensor{void *ptr;uint64_t bytes;int tier;};
static int g_n_gpus=1,called,q8,f32,resolves,seen_rows;static bool seen_stage,seen_pdl,dp=true,fail_launch=false,fail_fallback=false;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
static const void*cuda_resolve_weight_ptr(const void*p,uint64_t o,uint64_t,int,const char*){resolves++;return (const char*)p+o;}
static int cuda_q8_use_dp4a(){return dp;}static int cuda_decode_stream(){return 17;}
static int cudaGetLastError(){return fail_launch?1:0;}static int cuda_ok(int e,const char*){return !e;}
static int cuda_matmul_q8_0_preq_rows_exact(ds4_gpu_tensor*,const char*,const int8_t*,const float*,uint64_t,uint64_t,uint32_t,uint64_t){q8++;return !fail_fallback;}
static int ds4_gpu_matmul_f32_decode_rows_exact_tensor(ds4_gpu_tensor*,const void*,uint64_t,uint64_t,uint64_t,uint64_t,const ds4_gpu_tensor*,uint32_t){f32++;return !fail_fallback;}
'''+args+r'''
static void spy(unsigned r,bool stage,bool pdl,unsigned grid,unsigned threads,size_t panel,int stream,qwen_gdn_projection_args a){
 called++;seen_rows=r;seen_stage=stage;seen_pdl=pdl;
 assert(r==a.n_rows&&threads==256&&stream==17&&grid==(a.od[0]+3)/4+(a.od[1]+3)/4+96);
 assert(panel==(stage?4*80*34+16:0));
}
'''+overlap+'\n'+api+r'''
int main(){unsigned count=0;
 const char *envs[]={"DS4_QWEN4EXP_NO_WIDE_VERIFY","DS4_QWEN4EXP_WIDE_VERIFY_R2","DS4_QWEN4EXP_NO_ROW_TILE","DS4_QWEN4EXP_PAIR_LANES_R2","DS4_F32_NO_VECTOR_DECODE","DS4_QWEN4EXP_NO_GDN_PROJECTION_FUSION","DS4_QWEN4EXP_NO_GDN_PANEL"};
 ds4_gpu_tensor x{(void*)0x10000000,4*2560*4,0},q{(void*)0x20000000,4*80*36,0};
 ds4_gpu_tensor out[4];ds4_gpu_tensor*outs[4];const void*maps[4];uint64_t sizes[4],offs[4];
 for(unsigned bits=0;bits<128;bits++)for(unsigned rows:{1u,2u,3u,4u})for(unsigned dim:{516u,513u,10240u})for(unsigned offset:{0u,2u,4u})for(bool supported:{false,true}){
  for(unsigned j=0;j<7;j++)if(bits&(1u<<j))setenv(envs[j],"1",1);else unsetenv(envs[j]);
  uint64_t gate=dim==10240?6144:dim+2*(dim==513);uint64_t od[4]={dim,gate,48,48};
  for(unsigned i=0;i<4;i++){out[i]={(void*)(uintptr_t)(0x30000000u+i*0x10000000u),4*od[i]*4,0};outs[i]=&out[i];maps[i]=(void*)(uintptr_t)(0x80000000ull+i*0x10000000ull);offs[i]=i<2?offset:0;sizes[i]=offs[i]+od[i]*(i<2?80*34:2560*4);}
  called=q8=f32=resolves=0;dp=supported;fail_launch=false;fail_fallback=false;
  int ok=ds4_gpu_qwen4exp_gdn_projections_exact_tensor(outs,maps,sizes,offs,2560,dim,gate,&x,&q,0,rows*80*32,rows);
  bool fused=(rows<=2||(rows==3&&!(bits&3)))&&supported&&!(bits&60);
  assert(ok&&resolves==4);
  if(fused){assert(called==1&&q8==0&&f32==0&&seen_rows==(int)rows&&seen_pdl==(rows<=2));assert(seen_stage==(!(bits&64)&&offset!=2&&(rows!=3||((dim|gate)&3)==0)));}
  else assert(called==0&&q8==2&&f32==2);
  if(fused){called=q8=f32=0;fail_launch=true;assert(!ds4_gpu_qwen4exp_gdn_projections_exact_tensor(outs,maps,sizes,offs,2560,dim,gate,&x,&q,0,rows*80*32,rows));assert(called==1&&q8==0&&f32==0);}
  else {called=q8=f32=0;fail_fallback=true;assert(!ds4_gpu_qwen4exp_gdn_projections_exact_tensor(outs,maps,sizes,offs,2560,dim,gate,&x,&q,0,rows*80*32,rows));assert(called==0&&q8==1&&f32==0);}
  count++;
 }
 printf("PASS actual GDN wrapper: %u dispatch/control cases plus launch/fallback error short-circuit\n",count);
}
'''
with tempfile.TemporaryDirectory(prefix='gdn-r3-wrapper-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 subprocess.run(['c++','-O2','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test')],check=True)
