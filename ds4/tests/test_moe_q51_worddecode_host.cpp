/* Execute actual old/new MoE down CUDA bodies with scheduled host fibers.
 * Deferred async copies become visible only at wait; shuffles and barriers
 * block the appropriate threads. Not a GPU timing or native rounding test. */
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
struct alignas(16) uint4 { uint32_t x,y,z,w; };
struct alignas(8) uint2 { uint32_t x,y; };
struct alignas(16) int4 { int x,y,z,w; };
uint4 qw_down_panel[2048];
struct pending_copy { uint32_t dst; std::array<char,16> bytes; };
static std::vector<pending_copy> copies[256];
static bool committed[256];
static uint64_t copied, barriers;
static uint32_t __cvta_generic_to_shared(const void *p) {
    auto offset=(const char *)p-(const char *)qw_down_panel;
    need(offset>=0 && offset+16<=(long)sizeof(qw_down_panel),"panel bounds");
    return (uint32_t)offset;
}
static void qw_cpasync16(uint32_t dst,const void *src) {
    need(!committed[current],"copy appended to committed group");
    pending_copy c{};c.dst=dst;std::memcpy(c.bytes.data(),src,16);
    copies[current].push_back(c);copied+=16;
}
static void qw_cpasync_commit() {
    need(!committed[current],"two outstanding groups");
    committed[current]=true;
}
static void qw_cpasync_wait0() {
    for(const auto &c:copies[current])
        std::memcpy((char*)qw_down_panel+c.dst,c.bytes.data(),16);
    copies[current].clear();committed[current]=false;
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
#define __device__
#define __global__
#define __shared__
#define __forceinline__ inline
#define CUDA_QK_K 256
#define DS4_QWEN4EXP_WIDE_PAYLOAD 1
#include "ds4_qwen4exp_moe_types.h"
enum {
#define TYPE_ENUM(name,id) DS4_QWEN4EXP_TY_ ## name = id,
    DS4_QWEN4EXP_MOE_TYPES(TYPE_ENUM)
#undef TYPE_ENUM
};
#include "moe_down_bodies.inc"
static void cta(int nth,std::function<void()> f) {
    threads=nth;blockDim.x=nth;body=std::move(f);block_arrivals=0;
    std::fill(std::begin(arrivals),std::end(arrivals),0);
    std::memset(qw_down_panel,0xa5,sizeof(qw_down_panel));
    for(int i=0;i<threads;i++) {
        need(copies[i].empty()&&!committed[i],"unretired async group");
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
    for(int i=0;i<threads;i++)
        need(copies[i].empty()&&!committed[i],"kernel leaves async writes");
}
static uint32_t rng=0x296416u;
static uint32_t random_word() { rng=rng*1664525u+1013904223u;return rng; }
static unsigned cases=0;
template<int Type> static void trial(unsigned groups,unsigned nt,unsigned slots,
                                     unsigned pattern,unsigned rows,unsigned pad=0) {
    const unsigned experts=37, block_bytes=Type==8?34:24;
    const size_t row_bytes=groups*block_bytes+pad, expert_bytes=rows*row_bytes;
    const size_t bytes=experts*expert_bytes;
    std::vector<uint4> weights((bytes+15)/16);
    auto *w=(unsigned char*)weights.data();
    for(size_t i=0;i<bytes;i++) w[i]=(unsigned char)random_word();
    const uint16_t scales[]={0x1400,0xa801,0x8000,0,1,0x8001,0x4403};
    for(unsigned e=0;e<experts;e++) for(unsigned row=0;row<rows;row++)
        for(unsigned g=0;g<groups;g++) {
            size_t at=e*expert_bytes+row*row_bytes+g*block_bytes;
            uint16_t d=scales[(e+row+g+pattern)%7];std::memcpy(w+at,&d,2);
            if(Type==7) {d=scales[(e+2*row+g+3)%7];std::memcpy(w+at+2,&d,2);}
        }
    const auto saved_weights=weights;
    std::vector<int32_t> selected(nt*slots);
    for(unsigned t=0;t<nt;t++) for(unsigned s=0;s<slots;s++) {
        int v=pattern==0?(int)((t+s)%experts):
              pattern==1?(int)(s%3):(int)(random_word()%experts);
        if(pattern==3 && s%4==0) v=-1;
        if(pattern==3 && s%4==1) v=experts+7;
        if(pattern==4) v=-1;
        selected[t*slots+s]=v;
    }
    const size_t ng=(size_t)nt*slots*groups;
    std::vector<int4> packed(ng*2);
    auto *mq=(int8_t*)packed.data();
    std::vector<float> ms(ng);std::vector<int32_t> sums(ng);
    for(size_t g=0;g<ng;g++) {
        for(unsigned k=0;k<32;k++) {
            int v=(int8_t)random_word();
            if(pattern==4) v=0;
            mq[g*32+k]=v;sums[g]+=v;
        }
        const float mag[]={.0003f,1e-30f,1e3f,0.0f};
        ms[g]=mag[(g+pattern)%4]*(1.f+float(random_word()%997)/997.f);
    }
    const size_t outputs=nt*rows;
    std::vector<float> ref(outputs+8, -991.25f),got=ref,unstaged=ref;
    auto run=[&](unsigned mode) {
        for(unsigned b=0;b<rows/8;b++) for(unsigned t=0;t<(nt+1)/2;t++) {
            blockIdx.x=b;blockIdx.y=t;
            cta(256,[&] {
                if(mode==1) qwen4exp_moe_down_q_kernel<2,Type,true,true,true,true>(
                    got.data()+4,(const char*)w,selected.data(),mq,ms.data(),
                    sums.data(),expert_bytes,row_bytes,Type,groups,rows,nt,experts,slots);
                else if(mode==0) qwen4exp_moe_down_q_kernel<2,Type,true,true,true>(
                    ref.data()+4,(const char*)w,selected.data(),mq,ms.data(),
                    sums.data(),expert_bytes,row_bytes,Type,groups,rows,nt,experts,slots);
                else qwen4exp_moe_down_q_kernel<2,Type,true,false,false>(
                    unstaged.data()+4,(const char*)w,selected.data(),mq,ms.data(),
                    sums.data(),expert_bytes,row_bytes,Type,groups,rows,nt,experts,slots);
            });
        }
    };
    copied=0;run(0);const auto old_bytes=copied;
    copied=0;run(1);need(copied==old_bytes,"panel byte count changed");
    run(2);
    for(size_t i=0;i<outputs+8;i++) {
        if(std::memcmp(&ref[i],&got[i],4)||std::memcmp(&ref[i],&unstaged[i],4)) {
            std::fprintf(stderr,"type=%d groups=%u tokens=%u slots=%u pattern=%u row=%zu: %.9g %.9g %.9g\n",
                Type,groups,nt,slots,pattern,i,ref[i],got[i],unstaged[i]);
            fail("output bits differ");
        }
        if(i<4||i>=outputs+4) need(got[i]==-991.25f,"output canary");
        else need(std::isfinite(got[i])&&got[i]!=-991.25f,"missing output");
    }
    need(!std::memcmp(weights.data(),saved_weights.data(),bytes),"weight mutation");
    cases++;
}
int main() {
    (void)barriers;
    for(unsigned groups: {1u,16u,17u,20u,24u,31u,32u,33u,48u})
        for(unsigned nt: {1u,2u,3u}) for(unsigned pattern=0;pattern<5;pattern++) {
            unsigned slots=pattern==0?1:pattern==1?10:pattern==2?32:10;
            trial<8>(groups,nt,slots,pattern,16);
            trial<7>(groups,nt,slots,pattern,16);
        }
    // Full live output width, with every CTA and changing expert routes.
    trial<8>(20,2,10,2,2560);trial<7>(20,2,10,3,2560);
    // Panels stay aligned while padded rows exercise both word-load branches
    // and the two-byte-aligned Q5_1 decoder fallback.
    for(unsigned nt: {1u,2u}) {
        trial<7>(20,nt,10,2,16,2);trial<7>(20,nt,10,2,16,4);
        trial<8>(20,nt,10,2,16,2);
    }
    std::printf("MoE Q5_1 word decode: %u cases, staged/unstaged/worddecode bits and guards PASS\n",cases);
}
