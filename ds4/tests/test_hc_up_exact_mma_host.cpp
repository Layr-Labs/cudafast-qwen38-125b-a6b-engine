/* Actual candidate CUDA body on host fibers. The MMA emulator consumes the
 * actual per-lane fragments; independent int8 dots and 32-lane tree form the
 * reference. Native MMA rounding/performance still require GPU execution. */
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
static uint32_t mma_a[256][4],mma_b[256][2];
static void mma_m16n8k32_s8_q8(int32_t *d,const uint32_t *a,const uint32_t *b) {
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

static const unsigned char *weight_begin, *weight_end;
template<class T> static T __ldcs(const T *p) {
    const auto *b=(const unsigned char *)p;
    need(b>=weight_begin && b+sizeof(T)<=weight_end,"weight load out of slab");
    T v; std::memcpy(&v,p,sizeof v); return v;
}
using __half=uint16_t;
static uint16_t __ushort_as_half(uint16_t v) {return v;}
static float __half2float(uint16_t v) {return dev_f16_to_f32(v);}
static float ftz(float v) { return std::fpclassify(v)==FP_SUBNORMAL ? std::copysign(0.0f,v):v; }
static float hc_up_add(float a,float b) {return ftz(ftz(a)+ftz(b));}
static float hc_up_product(float w,float x,int d) {
    return ftz(std::fma(ftz(ftz(w)*ftz(x)),(float)d,0.0f));
}
#define __device__
#define __global__
#define __forceinline__ inline
#define QWEN4EXP_PDL_SYNC() ((void)0)
#include "hc_up_exact_bodies.inc"
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


static uint32_t rng=0x237ac196u;
static uint32_t next() {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void one(unsigned rows,unsigned channels,unsigned offset,unsigned pattern) {
    std::vector<uint32_t> weight((channels*340u+offset+63u)/4u,0x9185a366u);
    auto *w=(unsigned char *)weight.data()+offset;
    weight_begin=w; weight_end=w+channels*340u;
    uint16_t scales[]={0x2401,0x9800,0,0x8000,1,0x8001,0x3c00,0xbc00};
    for(unsigned n=0;n<channels;n++)for(unsigned g=0;g<10;g++) {
        auto *p=w+n*340u+g*34u;
        uint16_t sc=scales[(n+g+pattern)%8];std::memcpy(p,&sc,2);
        for(unsigned k=0;k<32;k++)p[k+2]=pattern==1?128:pattern==2?127:
            pattern==3?0:(unsigned char)next();
    }
    std::vector<int8_t> x(640);
    std::vector<float> xs(20);
    for(unsigned i=0;i<x.size();i++)x[i]=pattern==1?127:pattern==2?-128:
        pattern==4?0:(int8_t)next();
    const float sv[]={.003f,-.001f,0.0f,-0.0f,1e-38f,-1e-38f,1e10f,-1e-30f};
    for(unsigned i=0;i<20;i++)xs[i]=sv[(i+pattern)%8];
    std::vector<float> ref(2*channels+32,-771.5f),out=ref;
    const auto saved_weights=weight; const auto saved_x=x; const auto saved_xs=xs;
    std::vector<unsigned> blocks;
    if(channels==10240)blocks={0,159,319};
    else for(unsigned b=0;b<(channels+31)/32;b++)blocks.push_back(b);
    for(unsigned b:blocks) {
        blockIdx.x=b; cta(128,[&](){matmul_q8_hc_up_exact_mma_kernel(out.data(),w,x.data(),xs.data(),rows,channels);});
        for(unsigned r=0;r<rows;r++)for(unsigned n=b*32;n<std::min(channels,(b+1)*32);n++) {
            float lane[32]={};
            for(unsigned g=0;g<10;g++) {
                int dot=0;
                for(unsigned k=0;k<32;k++)dot+=(int)(int8_t)w[n*340+g*34+2+k]*(int)x[r*320+g*32+k];
                uint16_t h;std::memcpy(&h,w+n*340+g*34,2);
                const float sc=ftz(ftz(dev_f16_to_f32(h))*ftz(xs[r*10+g]));
                lane[g]=ftz(std::fma(sc,(float)dot,0.0f));
            }
            for(unsigned step=16;step;step/=2)
                for(unsigned i=0;i<step;i++)lane[i]=ftz(ftz(lane[i])+ftz(lane[i+step]));
            ref[r*channels+n]=lane[0];
        }
    }
    if(std::memcmp(out.data(),ref.data(),out.size()*sizeof(float))) {
        for(unsigned i=0;i<out.size();i++)if(std::memcmp(&out[i],&ref[i],4)) {
            std::fprintf(stderr,"rows%u n%u offset%u pattern%u index%u got%a expected%a\n",rows,channels,offset,pattern,i,out[i],ref[i]);break;
        }
        fail("exact output or guard mismatch");
    }
    need(weight==saved_weights && x==saved_x && !std::memcmp(xs.data(),saved_xs.data(),80),"input mutation");
}
int main() {
    unsigned cases=0;
    for(unsigned r:{1u,2u})for(unsigned n:{8u,32u,40u,96u,10240u})
        for(unsigned offset:{0u,2u})for(unsigned p=0;p<8;p++){one(r,n,offset,p);cases++;}
    std::printf("HC up exact MMA actual-body PASS: %u cases, guarded fragment loads, float bits and untouched inputs\n",cases);
}
