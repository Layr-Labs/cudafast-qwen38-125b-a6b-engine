/* Execute actual original/candidate down CUDA bodies with host fibers.
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
using std::min;

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
[[maybe_unused]] static int __dp4a(int a,int b,int c) {
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
static uint32_t mma_a[256][4],mma_b[256][2];
static void qw_mma_m16n8k32(int32_t *d,const uint32_t *a,const uint32_t *b) {
    std::memcpy(mma_a[current],a,16);std::memcpy(mma_b[current],b,8);
    warp_barrier(0xffffffffu);
    const unsigned base=current&~31u, lane=current&31u;
    for(unsigned i=0;i<4;i++) {
        const unsigned row=(lane>>2)+(i>=2?8:0),col=(lane&3)*2+(i&1);
        int sum=d[i];
        for(unsigned k=0;k<32;k++) {
            const unsigned alane=(row&7)*4+((k&15)>>2);
            const unsigned areg=(row>=8?1:0)+(k>=16?2:0);
            const unsigned blane=col*4+((k&15)>>2),breg=k>=16?1:0;
            const int av=(int8_t)(mma_a[base+alane][areg]>>((k&3)*8));
            const int bv=(int8_t)(mma_b[base+blane][breg]>>((k&3)*8));
            sum+=av*bv;
        }
        d[i]=sum;
    }
    warp_barrier(0xffffffffu);
}
#define __launch_bounds__(...)
#define __device__
#define __global__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
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
#include "down_tasks_bodies.inc"
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

static uint32_t rng=0x841c392bu;
static uint32_t random_word() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}

static unsigned cases;
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
static float from_bits(uint32_t u) { float x; std::memcpy(&x,&u,4); return x; }
template<int Type=8, bool Wide=true>
static void kernel_case(unsigned count,unsigned rows,unsigned groups,
                        unsigned pattern,bool compact) {
    const unsigned experts=3, pairs_count=count*experts;
    const unsigned block_bytes=Type==8?34u:24u;
    const size_t row_bytes=groups*block_bytes, expert_bytes=rows*row_bytes;
    std::vector<uint4> weights((experts*expert_bytes+31)/16);
    auto *w=(unsigned char *)weights.data() + ((pattern & 2u) ? 2u : 0u);
    const uint16_t hs[]={0,0x8000,1,0x8001,0x1400,0xa401,0x7bff};
    for(unsigned e=0;e<experts;e++) for(unsigned r=0;r<rows;r++)
        for(unsigned g=0;g<groups;g++) {
            const size_t at=e*expert_bytes+r*row_bytes+g*block_bytes;
            const uint16_t h=hs[(pattern+e+r+g)%7];
            std::memcpy(w+at,&h,2);
            for(unsigned k=2;k<block_bytes;k++) w[at+k]=(uint8_t)random_word();
            if(Type==7) {
                uint16_t bias=hs[(pattern+2*e+r+3*g+1)%7];
                std::memcpy(w+at+2,&bias,2);
            }
        }
    std::vector<int32_t> counts={int(count),0,int(pattern==4?min(count,1u):count)};
    std::vector<int32_t> offsets={0,int(count),int(count)};
    std::vector<int32_t> pairs(2*count),active={2,2,0};
    for(unsigned i=0;i<2*count;i++) pairs[i]=int((i+3)%(2*count));
    std::vector<int8_t> mq(pairs_count*groups*32u);
    std::vector<float> ms(pairs_count*groups);
    std::vector<int32_t> sums(pairs_count*groups);
    const uint32_t fs[]={0,0x80000000,1,0x80000001,0x0d800001,
                         0x3c000001,0x41000003,0x7f7fffff};
    for(unsigned p=0;p<pairs_count;p++) for(unsigned g=0;g<groups;g++) {
        const size_t at=p*groups+g;
        ms[at]=from_bits(fs[(p+g+pattern)%8]);
        for(unsigned k=0;k<32;k++) {
            int8_t q=pattern==3?0:(int8_t)random_word();
            if(pattern==4) q=k&1?-3:3;
            mq[at*32+k]=q;sums[at]+=q;
        }
    }
    std::vector<float> ref(pairs_count*rows+32,from_bits(0x428ae147)),got=ref;
    std::vector<int32_t> tasks={0};
    for(unsigned e: {2u,0u}) for(int32_t start=0;start<counts[e];start+=32) {
        tasks.push_back((int32_t)e);tasks.push_back(start);
    }
    tasks[0]=(int32_t)(tasks.size()-1)/2;
    auto run=[&](bool optimized) {
        const unsigned work=optimized?(unsigned)tasks[0]+2u:(compact?4:experts);
        for(blockIdx.y=0;blockIdx.y<work;blockIdx.y++)
            for(blockIdx.x=0;blockIdx.x<(rows+63)/64;blockIdx.x++) {
                cta(128,[&] {
#define CALL(B,P) qwen4exp_moe_down_mma_kernel<Type,Wide,B>( \
                    P.data(),(const char *)w,mq.data(),ms.data(),sums.data(), \
                    pairs.data(),counts.data(),offsets.data(),B?tasks.data():(compact?active.data():nullptr), \
                    expert_bytes,row_bytes,Type,groups,rows,pattern & 1u)
                    if(optimized) { CALL(true,got); } else { CALL(false,ref); }
#undef CALL
                });
            }
    };
    run(false);run(true);
    for(size_t i=0;i<ref.size();i++) {
        if(std::isnan(ref[i]) && std::isnan(got[i])) continue;
        if(bits(ref[i])!=bits(got[i])) {
            std::fprintf(stderr,"count=%u rows=%u groups=%u pattern=%u compact=%d index=%zu %08x != %08x\n",
                count,rows,groups,pattern,compact,i,bits(ref[i]),bits(got[i]));
            fail("partial/guard bit mismatch");
        }
    }
    ++cases;
}
int main() {
    for(unsigned count: {0u,1u,7u,8u,9u,16u,17u,31u,32u,33u,65u})
        kernel_case(count,64,20,count%5,false);
    for(unsigned g: {1u,3u,4u,5u,19u,21u,32u,36u})
        kernel_case(9,65,g,g%5,true);
    for(unsigned p=0;p<5;p++) kernel_case(17,7,20,p,true);
    for(unsigned count: {0u,1u,7u,8u,9u,16u,17u,31u,32u,33u,65u})
        kernel_case<7>(count,64,20,count%5,false);
    for(unsigned g: {1u,3u,4u,5u,19u,21u,32u,36u})
        kernel_case<7>(9,65,g,g%5,true);
    for(unsigned p=0;p<5;p++) kernel_case<7>(17,7,20,p,true);
    kernel_case<7,false>(17,64,20,1,true);
    kernel_case<7,false>(17,64,20,3,true);
    kernel_case<7>(257,7,4,4,true);
    kernel_case<8>(257,7,4,4,true);
    std::printf("bounded down tasks: %u actual-kernel cases PASS\n",cases);
}
