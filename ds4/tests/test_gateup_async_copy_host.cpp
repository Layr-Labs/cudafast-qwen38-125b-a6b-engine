/* Execute actual original/async-copy gate-up CUDA bodies with host fibers.
 * Shared barriers and shuffles synchronize the scheduled lanes. This checks
 * indexing and operation order, not native GPU rounding or performance. */
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
/* Copies may become visible at issue or at wait. Yielding after each issue
 * exposes a missing producer/consumer barrier to other lane fibers. */
struct Copy { void *dst; std::array<unsigned char,16> bytes; };
static std::vector<void *> shared_addresses;
static std::vector<Copy> pending[256], committed[256];
static uintptr_t weights_begin, weights_end, up_begin, up_end;
static size_t copies_issued;
static uint32_t __cvta_generic_to_shared(const void *p) {
    auto it=std::find(shared_addresses.begin(),shared_addresses.end(),p);
    if(it==shared_addresses.end()) { shared_addresses.push_back(const_cast<void*>(p));
        return (uint32_t)shared_addresses.size()-1; }
    return (uint32_t)(it-shared_addresses.begin());
}
static void qw_cpasync16(uint32_t dst,const void *src) {
    need(dst<shared_addresses.size(),"unregistered shared destination");
    uintptr_t addr=(uintptr_t)src;
    need((addr&15u)==0u,"unaligned global copy");
    need((addr>=weights_begin && addr+16<=weights_end)||(addr>=up_begin && addr+16<=up_end),"global copy out of bounds");
    Copy c{shared_addresses[dst],{}};std::memcpy(c.bytes.data(),src,16);
    need(((uintptr_t)c.dst&15u)==0u,"unaligned shared copy");
    if((copies_issued++&1u)==0u) std::memcpy(c.dst,c.bytes.data(),16);
    else pending[current].push_back(c);
    swapcontext(&lanes[current].ctx,&scheduler);
}
[[maybe_unused]] static void qw_cpasync_commit() {
    need(committed[current].empty(),"second outstanding copy group");
    committed[current]=std::move(pending[current]);pending[current].clear();
}
[[maybe_unused]] static void qw_cpasync_wait0() {
    need(pending[current].empty(),"wait before commit");
    for(const Copy &c:committed[current]) std::memcpy(c.dst,c.bytes.data(),16);
    committed[current].clear();
}
#define __launch_bounds__(...)

#define __device__
#define __global__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QW_GU_MAXNREG
#define QW_GU_COOP_ROWS 4u
#define QW_GU_COOP_ROW_U4 90u
#define QW_GU_COOP_GROUPS 80u
#define QW_GU_COOP_U4 (QW_GU_COOP_ROWS * QW_GU_COOP_ROW_U4)
#define __forceinline__ inline
#define CUDA_QK_K 256
#define DS4_QWEN4EXP_WIDE_PAYLOAD 1
#include "ds4_qwen4exp_moe_types.h"
enum {
#define TYPE_ENUM(name,id) DS4_QWEN4EXP_TY_ ## name = id,
    DS4_QWEN4EXP_MOE_TYPES(TYPE_ENUM)
#undef TYPE_ENUM
};
#include "gateup_bodies.inc"
static void cta(int nth,std::function<void()> f) {
    for(int i=0;i<nth;i++)need(pending[i].empty()&&committed[i].empty(),"unfinished async group");
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

static uint32_t rng=0x841c392bu;
static uint32_t random_word() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static unsigned kernel_checks;
static void kernel_case(unsigned nt,unsigned slots,unsigned rows,
                        unsigned pattern,bool compact) {
    const unsigned experts=11,groups=80,row_bytes=1440;
    const size_t expert_bytes=rows*row_bytes,weight_bytes=experts*expert_bytes;
    std::vector<uint4> gate((weight_bytes+15)/16),up=gate;
    const uint16_t scales[]={0x1400,0xa401,0x8000,0,1,0x8001,0x2c03};
    auto fill=[&](std::vector<uint4>& slab) {
        auto *p=(uint8_t *)slab.data();
        for(size_t i=0;i<weight_bytes;i++)p[i]=(uint8_t)random_word();
        for(unsigned e=0;e<experts;e++)for(unsigned row=0;row<rows;row++)
            for(unsigned sb=0;sb<10;sb++) {
                size_t at=e*expert_bytes+row*row_bytes+sb*144;
                uint16_t d=scales[(e+row+sb+pattern)%7];
                uint16_t m=scales[(e+2*row+3*sb+pattern+1)%7];
                std::memcpy(p+at,&d,2);std::memcpy(p+at+2,&m,2);
            }
    };
    fill(gate);fill(up);
    const auto gate_saved=gate,up_saved=up;
    weights_begin=(uintptr_t)gate.data();weights_end=weights_begin+weight_bytes;
    up_begin=(uintptr_t)up.data();up_end=up_begin+weight_bytes;
    std::vector<std::vector<int32_t>> lists(experts);
    std::vector<float> weights(nt*slots);
    for(unsigned t=0;t<nt;t++)for(unsigned slot=0;slot<slots;slot++) {
        const unsigned pair=t*slots+slot;
        weights[pair]=float(int(random_word()%401)-200)/800.f;
        if(pattern==4 || (pattern==3 && slot%3==0))continue;
        unsigned e=pattern==0?(t+slot)%experts:
                   pattern==1?slot%3:random_word()%experts;
        lists[e].push_back(pair);
    }
    std::vector<int32_t> pairs,counts(experts),offsets(experts),active(1,0);
    for(unsigned e=0;e<experts;e++) {
        offsets[e]=pairs.size();counts[e]=lists[e].size();
        pairs.insert(pairs.end(),lists[e].begin(),lists[e].end());
        if(counts[e])active.push_back(e);
    }
    active[0]=(int32_t)active.size()-1;
    std::reverse(active.begin()+1,active.end());
    std::vector<int4> x(nt*groups*2);
    std::vector<float> xs(nt*groups);
    std::vector<int32_t> sums(nt*groups);
    int8_t *xq=(int8_t *)x.data();
    const float magnitudes[]={0.00001f,0.0002f,1e-30f,0.0f};
    for(unsigned g=0;g<nt*groups;g++) {
        for(unsigned j=0;j<32;j++) {
            xq[g*32+j]=(int8_t)random_word();sums[g]+=xq[g*32+j];
        }
        xs[g]=magnitudes[(g+pattern)%4]*(1.f+float(random_word()%1024)/1024.f);
    }
    const auto x_saved=x;
    const unsigned stride=slots*rows+7;
    std::vector<float> ref(nt*stride+8,-991.25f),got=ref,direct=ref;
    auto run=[&](unsigned mode) {
        auto *out=(mode==0?ref.data():mode==1?got.data():direct.data())+4;
        for(unsigned b=0;b<(rows+3)/4;b++)for(unsigned e=0;e<experts;e++) {
            blockIdx.x=b;blockIdx.y=e;
            cta(256,[&] {
#define ARGS out,(const char *)gate.data(),(const char *)up.data(),xq, \
    xs.data(),sums.data(),pairs.data(),counts.data(),offsets.data(), \
    compact?active.data():nullptr,weights.data(),expert_bytes,row_bytes, \
    expert_bytes,row_bytes,12,12,groups,rows,stride,slots
                if(mode==0)gateup_parent<2,12,true,4,true>(ARGS);
                else if(mode==1)qwen4exp_moe_gateup_async_kernel<2,12,true,4,true>(ARGS);
                else gateup_parent<2,12,true,4,false>(ARGS);
#undef ARGS
            });
        }
    };
    run(0);run(1);run(2);
    for(size_t i=0;i<ref.size();i++) {
        if(std::memcmp(&ref[i],&got[i],4)||std::memcmp(&ref[i],&direct[i],4)) {
            std::fprintf(stderr,"nt=%u slots=%u rows=%u pattern=%u compact=%d index=%zu: %.9g %.9g %.9g\n",
                         nt,slots,rows,pattern,compact,i,ref[i],got[i],direct[i]);
            fail("kernel bits differ");
        }
        need(isfinite(got[i]),"nonfinite kernel output");
    }
    need(!std::memcmp(gate.data(),gate_saved.data(),weight_bytes),"gate mutation");
    need(!std::memcmp(up.data(),up_saved.data(),weight_bytes),"up mutation");
    need(!std::memcmp(x.data(),x_saved.data(),x.size()*sizeof(int4)),"activation mutation");
    for(unsigned t=0;t<nt;t++)for(unsigned j=slots*rows;j<stride;j++)
        need(got[4+t*stride+j]==-991.25f,"token output padding");
    for(unsigned j=0;j<4;j++) {
        need(got[j]==-991.25f,"leading canary");
        need(got[got.size()-1-j]==-991.25f,"trailing canary");
    }
    for(unsigned e=0;e<experts;e++)for(auto p:lists[e])for(unsigned row=0;row<rows;row++)
        need(got[4+(p/slots)*stride+(p%slots)*rows+row]!=-991.25f,"missing routed output");
    ++kernel_checks;
}

int main() {
    for(unsigned nt:{1u,2u,3u,5u})for(unsigned rows:{1u,3u,4u,5u,9u})
        for(unsigned pattern=0;pattern<5;pattern++)
            kernel_case(nt,pattern==0?1u:pattern==1?3u:10u,rows,pattern,pattern%2);
    kernel_case(2,10,640,2,true);
    kernel_case(3,10,5,1,false);
    std::printf("PASS: %u actual-kernel async-copy fiber cases\n",
                kernel_checks);
}
