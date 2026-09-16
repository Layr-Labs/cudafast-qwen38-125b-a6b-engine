"""Research-only native API/CUB execution with minimal ownership shims.
Usage: python3 native_api_standalone_gpu.py REPO OUTPUT.cu
No model, engine session, production weight resolver, or GB10 timing.
"""
from pathlib import Path
import re,sys
repo=Path(sys.argv[1]);main=(repo/'ds4/ds4_cuda.cu').read_text();native=(repo/'ds4/ds4_cuda_mtp_native.cuh').read_text();qwen=(repo/'ds4/ds4_cuda_qwen4exp.cuh').read_text()
def fn(s,m):
 a=s.index(m);e=s.index('{',a)+1;d=1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
code=r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line%d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define NEED(x) do{if(!(x)){fprintf(stderr,"assert line%d %s\n",__LINE__,#x);exit(2);}}while(0)
struct ds4_gpu_tensor{void*ptr;uint64_t bytes;int tier;};
static cudaStream_t stream;
static int g_n_gpus=1,g_cuda_no_top1=0;
static struct{int device_id;}g_gpu[1]={{0}};
static unsigned reads;
static cudaStream_t cuda_decode_stream(){return stream;}
static bool cuda_ok(cudaError_t e,const char*s){if(e!=cudaSuccess)fprintf(stderr,"%s: %s\n",s,cudaGetErrorString(e));return e==cudaSuccess;}
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t?t->tier:-1;}
static bool cuda_q8_use_dp4a(){return true;}
static const char*cuda_resolve_weight_ptr(const void*m,uint64_t o,uint64_t,int,const char*){return (const char*)m+o;}
static int ds4_gpu_tensor_read(const ds4_gpu_tensor*t,uint64_t o,void*p,uint64_t n){reads++;if(o>t->bytes||n>t->bytes-o)return 0;return cuda_ok(cudaMemcpyAsync(p,(char*)t->ptr+o,n,cudaMemcpyDeviceToHost,stream),"read")&&cuda_ok(cudaStreamSynchronize(stream),"read sync");}
'''
a=qwen.index('#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900');b=qwen.index('#endif',a)+len('#endif');code+=qwen[a:b]+'\n'
for name in ['DS4_Q8_QUANT_PDL_MAX_ROWS','DS4_Q8_QUANT_PDL_MAX_BLOCKS']:code+=re.search(r'#define '+name+r'[^\n]+',main)[0]+'\n'
for m in ['__device__ static float warp_sum_f32(', '__device__ __forceinline__ static uint32_t q8_top1_float_ordered_key(', '__device__ __forceinline__ static uint64_t q8_top1_pack_key(', '__device__ __forceinline__ static bool topk_score_better(', '__global__ static void indexer_top1_kernel(', '__global__ static void quantize_q8_0_f32_rows_warp_kernel(']:code+=fn(main,m)+'\n'
code+=native+'\n'
code+=r'''
static ds4_gpu_tensor alloc(uint64_t n){ds4_gpu_tensor t{nullptr,n,0};CK(cudaMalloc(&t.ptr,n+32));CK(cudaMemset(t.ptr,0xa5,n+32));return t;}
static void guard(const ds4_gpu_tensor&t){unsigned char b[32];CK(cudaMemcpy(b,(char*)t.ptr+t.bytes,32,cudaMemcpyDeviceToHost));for(auto v:b)NEED(v==0xa5);}
int main(int argc,char**){bool expect_sorted=argc>1;CK(cudaSetDevice(0));CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 constexpr uint32_t dim=2560,vocab=4096,prefix=3001,tail=276,width=prefix+tail,cap=2048;
 std::mt19937 rng(49001);unsigned cases=0;
 for(unsigned offset:{0u,2u})for(unsigned mode=0;mode<6;mode++)for(unsigned fused=0;fused<2;fused++){
  if(fused)unsetenv("DS4_MTP_NO_FUSED_SCREEN_KEYS");else setenv("DS4_MTP_NO_FUSED_SCREEN_KEYS","1",1);
  uint64_t wb=offset+(uint64_t)vocab*80*34;std::vector<unsigned char>w(wb);
  for(unsigned row=0;row<vocab;row++)for(unsigned g=0;g<80;g++){
   auto p=w.data()+offset+(size_t)row*80*34+g*34;uint16_t scale=0x3c00;
   if((mode==2||mode==3)&&row==0&&g==0)scale=mode==2?0x7e00:0x7c00;
   if(mode==4&&row==0&&g>=40)scale=0x7e00;
   if(mode==5&&row==1&&g>=40)scale=0x7e00;
   memcpy(p,&scale,2);for(unsigned j=0;j<32;j++)p[2+j]=mode==1?0:(unsigned char)((int)(rng()%15)-7);
  }
  void*dw;CK(cudaMalloc(&dw,wb));CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));
  uint64_t sb=0;uint32_t capacity=0;NEED(ds4_gpu_mtp_native_screen_init(width,&sb,&capacity)==1&&capacity==cap);
  auto x=alloc(dim*4),out=alloc(width*4),ids=alloc(cap*4),scratch=alloc(sb),winner=alloc(4);
  std::vector<float>activation(dim),oldscore(cap),newscore(cap);for(auto&v:activation)v=(int)(rng()%201)*.01f-1.f;
  CK(cudaMemcpy(x.ptr,activation.data(),dim*4,cudaMemcpyHostToDevice));
  reads=0;int status=ds4_gpu_mtp_native_screen(&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x);NEED(reads==1);
  bool invalid=mode==2||mode==3;NEED(status==(invalid?0:(int)cap));
  std::vector<uint32_t>oldids(cap),newids(cap);uint32_t expected=UINT32_MAX;
  if(!invalid){NEED(ds4_gpu_tensor_read(&ids,0,oldids.data(),cap*4));NEED(ds4_gpu_tensor_read(&out,0,oldscore.data(),cap*4));
   indexer_top1_kernel<<<1,1024,0,stream>>>((uint32_t*)winner.ptr,(float*)out.ptr,cap,1);CK(cudaGetLastError());NEED(ds4_gpu_mtp_native_map(&winner,&out,&ids,cap,vocab));NEED(ds4_gpu_tensor_read(&winner,0,&expected,4));}
  else{std::vector<unsigned char>before(cap*4);NEED(ds4_gpu_tensor_read(&out,0,before.data(),cap*4));for(auto v:before)NEED(v==0xa5);}
  reads=0;uint64_t flag_off=UINT64_MAX;NEED(ds4_gpu_mtp_native_propose_async(&winner,&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x,&flag_off)==cap);NEED(reads==0);
  uint32_t got=0,flag=0;NEED(ds4_gpu_tensor_read(&winner,0,&got,4));NEED(got==expected);NEED(flag_off==mtp_native_offsets(width).flag);NEED(ds4_gpu_tensor_read(&scratch,flag_off,&flag,4));NEED((flag!=0)==invalid);
  NEED(ds4_gpu_tensor_read(&ids,0,newids.data(),cap*4));NEED(ds4_gpu_tensor_read(&out,0,newscore.data(),cap*4));std::vector<unsigned char>seen(vocab,0);
  for(unsigned i=0;i<cap;i++){unsigned id=newids[i];NEED(id<vocab&&!seen[id]&&(id<prefix||id>=vocab-tail));seen[id]=1;if(expect_sorted&&i)NEED(newids[i]>newids[i-1]);}
  if(!invalid){for(unsigned i=0;i<cap;i++){auto it=std::lower_bound(oldids.begin(),oldids.end(),newids[i]);NEED(it!=oldids.end()&&*it==newids[i]);unsigned j=it-oldids.begin();NEED(!memcmp(&newscore[i],&oldscore[j],4));}NEED(newids[0]==0);if(!expect_sorted)NEED(newids[1]>=vocab-tail);}
  std::vector<float>after(dim);NEED(ds4_gpu_tensor_read(&x,0,after.data(),dim*4));NEED(!memcmp(after.data(),activation.data(),dim*4));std::vector<unsigned char>wafter(wb);CK(cudaMemcpy(wafter.data(),dw,wb,cudaMemcpyDeviceToHost));NEED(wafter==w);
  for(auto*t:{&x,&out,&ids,&scratch,&winner})guard(*t);
  // Whole-allocation alias rejection must precede all tensor writes.
  reads=0;NEED(ds4_gpu_mtp_native_propose_async(&winner,&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&out,&flag_off)==0&&reads==0);
  CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));NEED(ds4_gpu_mtp_native_propose_async(&winner,&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x,&flag_off)==0);cudaGraph_t graph;CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphDestroy(graph));
  for(auto*t:{&x,&out,&ids,&scratch,&winner})CK(cudaFree(t->ptr));CK(cudaFree(dw));cases++;
 }
 CK(cudaStreamDestroy(stream));printf("PASS %u actual native-header/CUB pipeline cases, %s async IDs, fused/unfused, offsets0/2, finite/NaN/Inf, score+winner parity, read counts, guards, alias/capture decline\n",cases,expect_sorted?"ascending":"score-ordered");
}
'''
Path(sys.argv[2]).write_text(code)
