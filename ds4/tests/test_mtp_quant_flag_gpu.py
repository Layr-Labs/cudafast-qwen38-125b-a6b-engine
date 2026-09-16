"""Native flag fusion: actual quantizer and synchronous native API/CUB checks.
Usage: python3 test_mtp_quant_flag_gpu.py REPO OUTPUT.cu
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
code+=r'''
static unsigned memset_calls;
static bool fail_memset, fail_launch_check;
static cudaError_t counted_memset(void*p,int v,size_t n,cudaStream_t st){memset_calls++;if(fail_memset)return cudaErrorInvalidValue;return cudaMemsetAsync(p,v,n,st);}
#define cudaMemsetAsync counted_memset
static cudaError_t counted_launch_error(){auto e=cudaGetLastError();if(fail_launch_check){fail_launch_check=false;return cudaErrorInvalidConfiguration;}return e;}
#define cudaGetLastError counted_launch_error
'''+native+'\n#undef cudaMemsetAsync\n#undef cudaGetLastError\n'

old=fn(main,'__global__ static void quantize_q8_0_f32_rows_warp_kernel(')
new=fn(native,'__global__ static void mtp_native_quantize_reset_kernel(')
new=new.replace('mtp_native_quantize_reset_kernel','quantize_q8_0_f32_rows_warp_kernel',1).replace('uint32_t n_rows, uint32_t *invalid) {','uint32_t n_rows) {',1).replace('\n    if (blockIdx.x == 0u && threadIdx.x == 0u) *invalid = 0u;','',1)
assert old==new, 'native quantizer arithmetic drift'
code+=r'''
static ds4_gpu_tensor alloc(uint64_t n){ds4_gpu_tensor t{nullptr,n,0};CK(cudaMalloc(&t.ptr,n+32));CK(cudaMemset(t.ptr,0xa5,n+32));return t;}
static void guard(const ds4_gpu_tensor&t){unsigned char b[32];CK(cudaMemcpy(b,(char*)t.ptr+t.bytes,32,cudaMemcpyDeviceToHost));for(auto v:b)NEED(v==0xa5);}
int main(){CK(cudaSetDevice(0));CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 constexpr uint32_t dim=2560,vocab=4096,prefix=3001,tail=276,width=prefix+tail,cap=2048;
 std::mt19937 rng(53001); unsigned quant_cases=0,api_cases=0;
 // Real quantizer byte/scale parity, including exceptional activations.
 auto q0=alloc(dim),q1=alloc(dim),s0=alloc(80*4),s1=alloc(80*4),x=alloc(dim*4),flag=alloc(4);
 for(unsigned mode=0;mode<8;mode++)for(unsigned trial=0;trial<16;trial++){
  std::vector<float> a(dim);for(auto&v:a)v=(int(rng()%20001)-10000)*.0001f;
  const uint32_t bits[]={0,0x80000000u,1u,0x00800000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7f7fffffu};
  for(unsigned j=0;j<dim;j+=31)memcpy(&a[j],&bits[mode],4);
  CK(cudaMemcpy(x.ptr,a.data(),dim*4,cudaMemcpyHostToDevice));CK(cudaMemsetAsync(flag.ptr,0xff,4,stream));
  quantize_q8_0_f32_rows_warp_kernel<<<10,256,0,stream>>>((int8_t*)q0.ptr,(float*)s0.ptr,(float*)x.ptr,dim,80,1);
  mtp_native_quantize_reset_kernel<<<10,256,0,stream>>>((int8_t*)q1.ptr,(float*)s1.ptr,(float*)x.ptr,dim,80,1,(uint32_t*)flag.ptr);CK(cudaGetLastError());
  std::vector<unsigned char>b0(dim),b1(dim);std::vector<uint32_t>sc0(80),sc1(80);uint32_t z=1;
  NEED(ds4_gpu_tensor_read(&q0,0,b0.data(),dim)&&ds4_gpu_tensor_read(&q1,0,b1.data(),dim)&&b0==b1);
  NEED(ds4_gpu_tensor_read(&s0,0,sc0.data(),320)&&ds4_gpu_tensor_read(&s1,0,sc1.data(),320)&&sc0==sc1);
  NEED(ds4_gpu_tensor_read(&flag,0,&z,4)&&z==0);for(auto*t:{&q0,&q1,&s0,&s1,&x,&flag})guard(*t);quant_cases++;
 }
 for(auto*t:{&q0,&q1,&s0,&s1,&x,&flag})CK(cudaFree(t->ptr));
 for(unsigned offset:{0u,2u})for(unsigned mode=0;mode<6;mode++)for(unsigned fused=0;fused<2;fused++){
  if(fused)unsetenv("DS4_MTP_NO_FUSED_SCREEN_KEYS");else setenv("DS4_MTP_NO_FUSED_SCREEN_KEYS","1",1);
  uint64_t wb=offset+(uint64_t)vocab*80*34;std::vector<unsigned char>w(wb);
  for(unsigned row=0;row<vocab;row++)for(unsigned g=0;g<80;g++){
   auto p=w.data()+offset+(size_t)row*80*34+g*34;uint16_t scale=0x3c00;
   if((mode==2||mode==3)&&row==0&&g==0)scale=mode==2?0x7e00:0x7c00;
   if(mode==4&&row==0&&g>=40)scale=0x7e00;if(mode==5&&row==1&&g>=40)scale=0x7e00;
   memcpy(p,&scale,2);for(unsigned j=0;j<32;j++)p[2+j]=mode==1?0:(unsigned char)((int)(rng()%15)-7);
  }
  void*dw;CK(cudaMalloc(&dw,wb));CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));
  uint64_t sb=0;uint32_t capacity=0;NEED(ds4_gpu_mtp_native_screen_init(width,&sb,&capacity)==1&&capacity==cap);
  auto x=alloc(dim*4),out=alloc(width*4),ids=alloc(cap*4),scratch=alloc(sb),winner=alloc(4);
  std::vector<float>activation(dim);for(auto&v:activation)v=(int)(rng()%201)*.01f-1.f;
  CK(cudaMemcpy(x.ptr,activation.data(),dim*4,cudaMemcpyHostToDevice));
  // Actual range guard rejects overlaps/overflow and accepts adjacent ranges.
  NEED(!mtp_native_key_range_disjoint(scratch.ptr,scratch.bytes,scratch.ptr,4));
  NEED(!mtp_native_key_range_disjoint((void*)(UINTPTR_MAX-3),8,x.ptr,4));
  NEED(mtp_native_key_range_disjoint(scratch.ptr,4,(char*)scratch.ptr+4,4));
  // Two phases on same scratch: exceptional initial model then entirely finite.
  for(unsigned phase=0;phase<2;phase++){
   if(phase){for(unsigned row=0;row<vocab;row++)for(unsigned g=0;g<80;g++){uint16_t scale=0x3c00;memcpy(w.data()+offset+(size_t)row*80*34+g*34,&scale,2);}CK(cudaMemcpy(dw,w.data(),wb,cudaMemcpyHostToDevice));}
   std::vector<uint32_t>oldids(cap),oldscore(cap),newids(cap),newscore(cap);std::vector<uint64_t>oldkeys(width),newkeys(width);
   uint32_t expected=UINT32_MAX,got=UINT32_MAX,oldflag,newflag;int statuses[2];
   for(unsigned arm=0;arm<2;arm++){
    if(!arm)setenv("DS4_MTP_NO_QUANT_FLAG_FUSION","1",1);else unsetenv("DS4_MTP_NO_QUANT_FLAG_FUSION");
    CK(cudaMemsetAsync((char*)scratch.ptr+mtp_native_offsets(width).flag,0xff,4,stream));
    memset_calls=0;reads=0;statuses[arm]=ds4_gpu_mtp_native_screen(&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x);NEED(reads==1&&memset_calls==(arm?0u:1u));
    bool invalid=!phase&&(mode==2||mode==3);NEED(statuses[arm]==(invalid?0:(int)cap));
    auto layout=mtp_native_offsets(width);NEED(ds4_gpu_tensor_read(&scratch,layout.flag,arm?&newflag:&oldflag,4));
    NEED(ds4_gpu_tensor_read(&scratch,layout.key_in,(arm?newkeys:oldkeys).data(),width*8));
    if(!invalid){NEED(ds4_gpu_tensor_read(&ids,0,(arm?newids:oldids).data(),cap*4));NEED(ds4_gpu_tensor_read(&out,0,(arm?newscore:oldscore).data(),cap*4));
     indexer_top1_kernel<<<1,1024,0,stream>>>((uint32_t*)winner.ptr,(float*)out.ptr,cap,1);CK(cudaGetLastError());NEED(ds4_gpu_mtp_native_map(&winner,&out,&ids,cap,vocab));NEED(ds4_gpu_tensor_read(&winner,0,arm?&got:&expected,4));}
   }
   NEED(oldflag==newflag&&oldkeys==newkeys&&oldids==newids&&oldscore==newscore&&expected==got);api_cases++;
  }
  // Test-only runtime-error facade; no device fault is created. The real
  // quantizer may run, but the injected launch-status failure must abort
  // before screening/readback. Old-arm memset refusal must do likewise.
  if(offset==0&&mode==0&&fused==0){
   setenv("DS4_MTP_NO_QUANT_FLAG_FUSION","1",1);fail_memset=true;reads=0;
   NEED(ds4_gpu_mtp_native_screen(&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x)==-1&&reads==0);fail_memset=false;
   for(unsigned arm=0;arm<2;arm++){
    if(arm)unsetenv("DS4_MTP_NO_QUANT_FLAG_FUSION");else setenv("DS4_MTP_NO_QUANT_FLAG_FUSION","1",1);
    fail_launch_check=true;reads=0;NEED(ds4_gpu_mtp_native_screen(&out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x)==-1&&reads==0&&!fail_launch_check);CK(cudaStreamSynchronize(stream));
   }
  }
  // Alias output into dead CUB temporary storage: valid full API operation,
  // but full-allocation guard must keep the old reset/quantizer sequence.
  auto layout=mtp_native_offsets(width);NEED(scratch.bytes-layout.temporary>=cap*4);
  ds4_gpu_tensor alias_out{(char*)scratch.ptr+layout.temporary,cap*4,0};
  unsetenv("DS4_MTP_NO_QUANT_FLAG_FUSION");memset_calls=0;reads=0;
  NEED(ds4_gpu_mtp_native_screen(&alias_out,&ids,&scratch,dw,wb,offset,dim,vocab,prefix,tail,&x)==cap);
  NEED(memset_calls==1&&reads==1);CK(cudaStreamSynchronize(stream));
  std::vector<float>after(dim);NEED(ds4_gpu_tensor_read(&x,0,after.data(),dim*4));NEED(!memcmp(after.data(),activation.data(),dim*4));std::vector<unsigned char>wa(wb);CK(cudaMemcpy(wa.data(),dw,wb,cudaMemcpyDeviceToHost));NEED(wa==w);
  for(auto*t:{&x,&out,&ids,&scratch,&winner}){guard(*t);CK(cudaFree(t->ptr));}CK(cudaFree(dw));
 }
 CK(cudaStreamDestroy(stream));printf("PASS %u actual quantizer byte/scale/flag cases; %u synchronous native API/CUB A/B cases, nonfinite-to-finite reset, keys/sorted IDs/logit bits/winner/canaries\n",quant_cases,api_cases);
}
'''
Path(sys.argv[2]).write_text(code)
