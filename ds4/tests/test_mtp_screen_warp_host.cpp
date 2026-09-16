/* Actual one-warp CUDA body with host fibers and packed DP4A emulation.
 * Independent raw scalar dots, float tree and keys are the oracle. */
#include <ucontext.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>
using std::isfinite;

struct dim { uint32_t x=0, y=0, z=0; };
static dim threadIdx, blockIdx, blockDim;
struct lane_state {
    ucontext_t ctx{};
    std::vector<char> stack = std::vector<char>(16384);
    int state=0;
};
static lane_state lanes[256];
static ucontext_t scheduler;
static int current, threads, block_arrivals;
static uint32_t arrivals[8], masks[8], values[256];
static std::function<void()> body;
[[noreturn]] static void fail(const char *msg) {
    std::fprintf(stderr,"FAIL: %s (block %u thread %d)\n",msg,blockIdx.x,current);
    std::exit(1);
}
static void need(bool ok,const char *msg) { if(!ok) fail(msg); }
static void fiber_entry() {
    body(); lanes[current].state=-1;
    swapcontext(&lanes[current].ctx,&scheduler); std::abort();
}
[[maybe_unused]] static void __syncthreads() {
    lanes[current].state=9;
    if(++block_arrivals==threads) {
        for(int i=0;i<threads;i++) {
            need(lanes[i].state==9,"divergent block barrier");
            lanes[i].state=0;
        }
        block_arrivals=0;
    }
    swapcontext(&lanes[current].ctx,&scheduler);
}
static void warp_barrier(uint32_t mask) {
    int w=current/32;
    need(mask&(1u<<(current&31)),"caller excluded from shuffle mask");
    if(!arrivals[w]) masks[w]=mask;
    need(masks[w]==mask,"inconsistent shuffle masks");
    lanes[current].state=w+1;
    arrivals[w]|=1u<<(current&31);
    if(arrivals[w]==mask) {
        for(int i=w*32;i<(w+1)*32;i++) if(mask&(1u<<(i&31))) {
            need(lanes[i].state==w+1,"divergent warp barrier");
            lanes[i].state=0;
        }
        arrivals[w]=0;
    }
    swapcontext(&lanes[current].ctx,&scheduler);
}
template<class T> static T shuffle(uint32_t mask,T v,int from) {
    need(mask&(1u<<from),"read from inactive shuffle lane");
    std::memcpy(&values[current],&v,4);
    warp_barrier(mask);
    T result;
    std::memcpy(&result,&values[(current&~31)+from],4);
    warp_barrier(mask);
    return result;
}
template<class T> static T __shfl_xor_sync(uint32_t mask,T v,int d) {
    return shuffle(mask,v,(current&31)^d);
}
template<class T> static T __shfl_down_sync(uint32_t mask,T v,int d,int width=32) {
    int lane=current&31;
    return shuffle(mask,v,(lane%width)+d<width?lane+d:lane);
}
template<class T> static T __shfl_sync(uint32_t mask,T v,int lane) {
    return shuffle(mask,v,lane);
}
static int __dp4a(int a,int b,int c) {
    for(int i=0;i<4;i++) c+=(int)(int8_t)((uint32_t)a>>(8*i))*
                                (int)(int8_t)((uint32_t)b>>(8*i));
    return c;
}
static uint32_t __funnelshift_r(uint32_t lo,uint32_t hi,uint32_t s) {
    s&=31u;return s?(lo>>s)|(hi<<(32-s)):lo;
}
static float dev_f16_to_f32(uint16_t bits) {
    _Float16 h;std::memcpy(&h,&bits,2);return (float)h;
}
using __half=uint16_t;
static float __half2float(uint16_t h){return dev_f16_to_f32(h);}
static float ftz(float v){return std::fpclassify(v)==FP_SUBNORMAL?std::copysign(0.0f,v):v;}
static float mtp_screen_add(float a,float b){return ftz(ftz(a)+ftz(b));}
static float mtp_screen_product(float w,float x,int d){return ftz(std::fma(ftz(ftz(w)*ftz(x)),(float)d,0.0f));}
static uint32_t atomicOr(uint32_t *p,uint32_t v){auto old=*p;*p|=v;return old;}
static uint64_t q8_top1_pack_key(float f,uint32_t id){uint32_t u;std::memcpy(&u,&f,4);return ((uint64_t)((u&0x80000000u)?~u:u^0x80000000u)<<32)|(UINT32_MAX-id);}
static const unsigned char *weight_begin, *weight_end;
static const unsigned char *warp_block_begin,*warp_block_end;
static void host_warp_bounds(const unsigned char *block) {
 need(block>=weight_begin && block+34u<=weight_end,"warp group outside tensor");
 need(((size_t)(block-weight_begin)%2720u)%34u==0u,"warp group layout");
 need((size_t)(block-weight_begin)%2720u<816u,"warp group outside original prefix");
 warp_block_begin=block;warp_block_end=block+34u;
}
template<class T> static T warp_read(const void *ptr) {
 const auto *p=(const unsigned char *)ptr;
 need(p>=warp_block_begin && p+sizeof(T)<=warp_block_end,"warp group read overrun");
 need((uintptr_t)p%alignof(T)==0u,"warp load alignment");
 T v;std::memcpy(&v,p,sizeof v);return v;
}
static uint32_t mtp_warp_read4(const void *p){return warp_read<uint32_t>(p);}
static uint16_t mtp_warp_read2(const void *p){return warp_read<uint16_t>(p);}
#define __device__
#define __global__
#define __forceinline__ inline
#include "mtp_screen_warp_bodies.inc"
static void cta(int nth,std::function<void()> f) {
    threads=nth;blockDim.x=nth;body=std::move(f);block_arrivals=0;
    std::fill(std::begin(arrivals),std::end(arrivals),0);
    for(int i=0;i<threads;i++) {
        auto &l=lanes[i];getcontext(&l.ctx);
        l.ctx.uc_stack.ss_sp=l.stack.data();l.ctx.uc_stack.ss_size=l.stack.size();
        l.ctx.uc_link=&scheduler;l.state=0;
        makecontext(&l.ctx,fiber_entry,0);
    }
    int cursor=0;
    for(;;) {
        bool alive=false,ran=false;
        for(int j=0;j<threads;j++) {
            int i=(cursor+j)%threads;
            if(lanes[i].state!=-1) alive=true;
            if(lanes[i].state!=0) continue;
            current=i;threadIdx.x=i;
            swapcontext(&scheduler,&lanes[i].ctx);
            cursor=(i+73)%threads;ran=true;break;
        }
        if(!alive) break;
        need(ran,"block deadlock");
    }

}



static uint32_t rng=0x331729a5u;
static uint32_t next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void one(unsigned prefix,unsigned tail,unsigned offset,unsigned pattern){
 const unsigned width=prefix+tail,vocab=prefix>1000?248320:width+19;
 const size_t bytes=(size_t)vocab*2720;
 auto *raw=(unsigned char *)std::malloc(bytes+offset+64);need(raw!=nullptr,"allocation");
 auto *w=raw+offset;
 weight_begin=w;weight_end=w+bytes;
 /* Deterministic gap bytes: a wrong logical-row load cannot be detected
  * accidentally through unrelated allocator contents. */
 std::memset(w+(size_t)prefix*2720,0,(size_t)tail*2720);
 std::vector<unsigned> blocks;
 if(width>1024){
  const unsigned last=(width-1u)/4u;
  for(unsigned anchor:{0u,width/8u,prefix/4u,last})
   for(unsigned j=0;j<4u;j++)blocks.push_back(std::min(anchor,last-3u)+j);
 } else for(unsigned b=0;b<(width+3)/4;b++)blocks.push_back(b);
 std::sort(blocks.begin(),blocks.end());blocks.erase(std::unique(blocks.begin(),blocks.end()),blocks.end());
 std::vector<unsigned> touched;
 const uint16_t halfs[]={0x2401,0x9800,0,0x8000,1,0x8001,0x3c00,0xbc00};
 std::vector<uint8_t> saved;
 for(unsigned b:blocks)for(unsigned row=b*4;row<std::min(width,(b+1)*4);row++){
  unsigned id=row<prefix?row:vocab-tail+(row-prefix);touched.push_back(id);
  auto *p=w+(size_t)id*2720;
  for(unsigned g=0;g<80;g++){
   uint16_t h=halfs[(id+g+pattern)%8];
   if(pattern==8 || (pattern==9 && id==vocab-1))h=0x7c00;
   if(pattern==10 && id==0)h=0x7e00;
   std::memcpy(p+g*34,&h,2);
   for(unsigned k=0;k<32;k++)p[g*34+2+k]=pattern==1?128:pattern==2?127:pattern==3?0:(uint8_t)next();
  }
  saved.insert(saved.end(),p,p+2720);
 }
 std::vector<int8_t> x(2560);std::vector<float> xs(80);
 for(unsigned i=0;i<2560;i++)x[i]=pattern==1?127:pattern==2?-128:pattern==4?0:(int8_t)next();
 const float fs[]={.003f,-.001f,0,-0.0f,1e-38f,-1e-38f,1e10f,-1e-30f};
 for(unsigned i=0;i<80;i++)xs[i]=fs[(i+pattern)%8];
 const auto sx=x;const auto ss=xs;
 constexpr uint64_t canary=0x6192baeb44710119ull;
 std::vector<uint64_t> out(width+32,canary),expected=out;
 uint32_t invalid=0,reference_invalid=0;
 for(unsigned b:blocks){
  blockIdx.x=b;cta(128,[&](){mtp_native_screen_warp_kernel(out.data(),&invalid,w,x.data(),xs.data(),width,vocab,prefix,tail);});
  for(unsigned row=b*4;row<std::min(width,(b+1)*4);row++){
   unsigned id=row<prefix?row:vocab-tail+(row-prefix);
   float lane[32]={};
   for(unsigned g=0;g<24;g++){
    auto *p=w+(size_t)id*2720+g*34;int dot=0;
    for(unsigned k=0;k<32;k++)dot+=(int)(int8_t)p[2+k]*(int)x[g*32+k];
    uint16_t h;std::memcpy(&h,p,2);float sc=ftz(ftz(dev_f16_to_f32(h))*ftz(xs[g]));
    lane[g]=ftz(std::fma(sc,(float)dot,0.0f));
   }
   for(unsigned step=16;step;step/=2)for(unsigned i=0;i<step;i++)lane[i]=ftz(ftz(lane[i])+ftz(lane[i+step]));
   float v=lane[0];if(!std::isfinite(v))reference_invalid|=1;
   uint32_t bits;float normalized=v==0?0:v;std::memcpy(&bits,&normalized,4);
   uint32_t ordered=bits>>31?~bits:bits^0x80000000u;
   expected[row]=(!id||row>=prefix)?UINT64_MAX-id:((uint64_t)ordered<<32)|(UINT32_MAX-id);
  }
 }
 need(invalid==reference_invalid,"invalid flag mismatch");


 if(!invalid && out!=expected){
  for(unsigned i=0;i<out.size();i++)if(out[i]!=expected[i]){
   std::fprintf(stderr,"prefix%u tail%u offset%u pattern%u at%u got%llx expected%llx\n",prefix,tail,offset,pattern,i,(unsigned long long)out[i],(unsigned long long)expected[i]);break;
  }
  fail("coarse key mismatch");
 }
 for(unsigned i=width;i<out.size();i++)need(out[i]==canary,"output guard");
 size_t at=0;for(unsigned id:touched){need(!std::memcmp(w+(size_t)id*2720,saved.data()+at,2720),"weight mutation");at+=2720;}
 need(x==sx&&!std::memcmp(xs.data(),ss.data(),320),"input mutation");std::free(raw);
}
int main(){unsigned cases=0;
 for(unsigned p:{1u,17u,31u,32u,257u,98308u})for(unsigned t:{1u,7u,276u})
  for(unsigned offset:{0u,2u})for(unsigned pattern=0;pattern<11;pattern++){one(p,t,offset,pattern);cases++;}
 std::printf("MTP one-warp screen actual-body PASS: %u cases; exact keys/flags, bounded original group loads and both alignments\n",cases);
}
