"""Actual additive API and Qwen helpers with counted host runtime facades.
Preserves wrapper statements; the only CUDA launch is replaced with a spy.
"""
from pathlib import Path
import re,subprocess,tempfile
root=Path(__file__).resolve().parents[1]
s=(root/'ds4_cuda.cu').read_text();h=(root/'ds4_qwen4exp_matmul.h').read_text()
def body(s,mark):
 a=s.index(mark);b=s.index('{',a);e=b+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
api=body(s,'extern "C" int ds4_gpu_qwen4exp_indexer_k_bf16_r2(')
api,n=re.subn(r'QWEN4EXP_LAUNCH_PDL\(qwen4exp_indexer_k_bf16_r2_kernel,', 'spy(',api);assert n==1
# Retained source order is mandatory; this is not PDL runtime validation.
k=body(s,'__global__ static void qwen4exp_indexer_k_bf16_r2_kernel(')
assert '__restrict__' not in k
assert k.index('wrow[lane]') < k.index('QWEN4EXP_PDL_SYNC()') < k.index('x[lane]') < k.index('x[2560u + lane]')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <initializer_list>
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int tier;};
struct dim3{unsigned x,y,z;dim3(unsigned a,unsigned b,unsigned c):x(a),y(b),z(c){}};
static int g_cublas_ready=1,resolves,launches,oldcalls,grids,views,frees,fail_resolve,fail_launch,fail_old;static std::vector<unsigned>oldrows;
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->tier;}
static const char*cuda_resolve_weight_ptr(const void*p,uint64_t off,uint64_t bytes,int,const char*){resolves++;assert(bytes==128*2560*2);return fail_resolve?nullptr:(const char*)((uintptr_t)p+off);}
static int cuda_decode_stream(){return 7;}static int cudaGetLastError(){return fail_launch;}static int cuda_ok(int e,const char*){return !e;}
static void spy(dim3 g,unsigned block,unsigned smem,int stream,float*,const uint16_t*,const float*){assert(g.x==32&&g.y==1&&g.z==1&&block==128&&!smem&&stream==7);launches++;}
static int ds4_gpu_glm53_matmul_bf16(ds4_gpu_tensor*,const void*,uint64_t,uint64_t,unsigned,unsigned,const ds4_gpu_tensor*,unsigned rows){oldcalls++;oldrows.push_back(rows);return !fail_old;}
static int ds4_gpu_qwen4exp_bf16_prefill_exact_tensor(ds4_gpu_tensor*,const void*,uint64_t,uint64_t,unsigned,unsigned,const ds4_gpu_tensor*,unsigned){grids++;return 1;}
static ds4_gpu_tensor*ds4_gpu_tensor_view(const ds4_gpu_tensor*t,uint64_t off,uint64_t bytes){views++;assert(off<=t->bytes&&bytes<=t->bytes-off);return new ds4_gpu_tensor{(void*)((uintptr_t)t->ptr+off),bytes,t->tier};}
static void ds4_gpu_tensor_free(ds4_gpu_tensor*t){frees++;delete t;}
#define DS4_QWEN4EXP_BF16_DECODE_ROWS 8u
'''+api+'\n'+body(h,'static inline int ds4_qwen4exp_matmul_bf16(')+'\n'+body(h,'static inline int ds4_qwen4exp_indexer_k_bf16(')+r'''
int main(){unsigned cases=0;ds4_gpu_tensor out{(void*)0x10000000,1024,0},x{(void*)0x20000000,20480,0};const void*map=(void*)0x40000000;const uint64_t wb=128*2560*2;
 auto reset=[&](){out={(void*)0x10000000,1024,0};x={(void*)0x20000000,20480,0};resolves=launches=oldcalls=grids=views=frees=fail_resolve=fail_launch=fail_old=0;g_cublas_ready=1;oldrows.clear();unsetenv("DS4_QWEN4EXP_NO_INDEXER_K_R2");unsetenv("DS4_QWEN4EXP_NO_BF16_PREFILL_GRID");};
 auto run=[&](uint64_t size,uint64_t off){cases++;return ds4_gpu_qwen4exp_indexer_k_bf16_r2(&out,map,size,off,&x);};
 for(unsigned off:{0u,2u}){reset();assert(run(wb+off,off));assert(launches==1&&resolves==1&&!oldcalls);}
 for(unsigned bad=0;bad<9;bad++){reset();uint64_t off=0,size=wb;switch(bad){case 0:out.ptr=nullptr;break;case 1:x.ptr=nullptr;break;case 2:out.bytes=1023;break;case 3:x.bytes=20479;break;case 4:g_cublas_ready=0;break;case 5:off=wb+1;break;case 6:size=wb-1;break;case 7:fail_resolve=1;break;case 8:off=UINT64_MAX;break;}assert(!run(size,off));assert(!launches&&!oldcalls);}
 reset();assert(!ds4_gpu_qwen4exp_indexer_k_bf16_r2(nullptr,map,wb,0,&x));assert(!ds4_gpu_qwen4exp_indexer_k_bf16_r2(&out,nullptr,wb,0,&x));assert(!ds4_gpu_qwen4exp_indexer_k_bf16_r2(&out,map,wb,0,nullptr));
 for(unsigned alias=0;alias<4;alias++){reset();if(alias==0)out.ptr=(char*)x.ptr+4;if(alias==1)out.ptr=(void*)((uintptr_t)map+2);if(alias==2)out.ptr=(void*)(UINTPTR_MAX-7);if(alias==3)x.bytes=UINT64_MAX;assert(run(wb,0));assert(oldcalls==1&&!launches);}
 reset();out.ptr=(char*)x.ptr+x.bytes;assert(run(wb,0));assert(launches==1&&!oldcalls);
 reset();setenv("DS4_QWEN4EXP_NO_INDEXER_K_R2","1",1);assert(run(wb,0));assert(oldcalls==1&&!launches);fail_old=1;assert(!run(wb,0));assert(oldcalls==2&&!launches);
 reset();fail_launch=1;assert(!run(wb,0));assert(launches==1&&!oldcalls);
 for(unsigned rows:{0u,1u,2u,3u,8u,9u,16u,65536u})for(unsigned in:{2560u,2561u})for(unsigned od:{128u,129u})for(bool nogrid:{false,true}){
  reset();if(nogrid)setenv("DS4_QWEN4EXP_NO_BF16_PREFILL_GRID","1",1);out.bytes=(uint64_t)rows*od*4;x.bytes=(uint64_t)rows*in*4;
  cases++;assert(ds4_qwen4exp_indexer_k_bf16(&out,map,wb,0,in,od,&x,rows));
#if !defined(DS4_NO_GPU) && !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
  const bool native=rows==2&&in==2560&&od==128;
  const bool grid=!native&&rows>8&&rows<=65535&&!nogrid;
#else
  const bool native=false,grid=false;
#endif
  if(native){assert(launches==1&&oldcalls==0&&grids==0);}
  else if(grid){assert(grids==1&&oldcalls==0&&launches==0);}
  else{assert(!grids&&!launches);assert(oldcalls==(rows<=8?1:(rows+7)/8));for(unsigned n:oldrows)assert(n<=8);assert(views==frees&&views==(rows<=8?0:2*oldcalls));}
 }
 printf("PASS indexer-K actual API/helper controls: %u cases; backend guards, exact wide policy, aliases and errors\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='indexer-k-r2-controls-') as t:
 p=Path(t);(p/'test.cpp').write_text(source)
 for backend in [None,'DS4_NO_GPU','__APPLE__','DS4_ROCM_BUILD']:
  subprocess.run(['c++','-O2','-std=c++17']+(['-D'+backend] if backend else [])+[str(p/'test.cpp'),'-o',str(p/'test')],check=True)
  subprocess.run([str(p/'test')],check=True)
