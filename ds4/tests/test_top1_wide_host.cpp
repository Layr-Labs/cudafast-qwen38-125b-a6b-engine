/* Execute the production top-1 kernels with scheduled host lanes.
 * This checks comparison/index semantics and barriers, not GPU speed. */
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
#include <map>
#include <vector>
using std::isfinite;

struct dim { uint32_t x=0, y=0, z=0; };
static dim threadIdx, blockIdx, blockDim;
struct lane_state {
    ucontext_t ctx{};
    std::vector<char> stack = std::vector<char>(16384);
    int state=0;
};
static lane_state lanes[1024];
static ucontext_t scheduler;
static int current, threads, block_arrivals;
static uint32_t arrivals[32], masks[32], values[1024];
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
template<class T> static T __shfl_down_sync(uint32_t mask,T v,int d,int width=32) {
    int lane=current&31;return shuffle(mask,v,(lane%width)+d<width?lane+d:lane);
}
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#include "top1_bodies.inc"
static void cta(int nth,std::function<void()> f) {
    threads=nth;blockDim.x=nth;body=std::move(f);block_arrivals=0;
    std::fill(std::begin(arrivals),std::end(arrivals),0);
    for(int i=0;i<threads;i++) {
        auto &l=lanes[i];getcontext(&l.ctx);
        l.ctx.uc_stack.ss_sp=l.stack.data();l.ctx.uc_stack.ss_size=l.stack.size();
        l.ctx.uc_link=&scheduler;l.state=0;makecontext(&l.ctx,fiber_entry,0);
    }
    int cursor=0;
    for(;;) {
        bool alive=false,ran=false;
        for(int j=0;j<threads;j++) {
            int i=(cursor+j)%threads;
            if(lanes[i].state!=-1) alive=true;
            if(lanes[i].state!=0) continue;
            current=i;threadIdx.x=i;swapcontext(&scheduler,&lanes[i].ctx);
            cursor=(i+73)%threads;ran=true;break;
        }
        if(!alive) break;need(ran,"block deadlock");
    }
}
static uint32_t rng=0x4519defu;
static uint32_t rnd(){rng=rng*1664525u+1013904223u;return rng;}
static unsigned cases;
static void trial(unsigned n,unsigned rows,unsigned pattern) {
    std::vector<float> x((size_t)n*rows);
    for(unsigned t=0;t<rows;t++) {
        for(unsigned i=0;i<n;i++) {
            float v=float(int(rnd()%19999)-9999)*.125f;
            if(pattern==1) v=-INFINITY;
            if(pattern==2) v=NAN;
            if(pattern==3 || pattern==4 || pattern==5) v=-1.f;
            if(pattern==6) v=std::ldexp(float(int(rnd()%7)-3),-148);
            if(pattern==7) v=i%2?NAN:-INFINITY;
            x[(size_t)t*n+i]=v;
        }
        if(pattern==0 || pattern==5) x[(size_t)t*n+n-1-t]=INFINITY;
        if(pattern==3 || pattern==4) {
            unsigned a=std::min(n-1,4095u),b=std::min(n-1,4096u);
            x[(size_t)t*n+a]=pattern==4?-0.f:11.f;
            x[(size_t)t*n+b]=pattern==4?0.f:11.f;
            x[(size_t)t*n+n-1]=pattern==4?0.f:11.f;
        }
    }
    const auto saved=x;
    std::vector<uint32_t> expected(rows),old(rows+8,0xdeadbeefu),got=old;
    for(unsigned t=0;t<rows;t++) {
        float best=-INFINITY;unsigned ix=0;
        for(unsigned i=0;i<n;i++) if(x[(size_t)t*n+i]>best) {best=x[(size_t)t*n+i];ix=i;}
        expected[t]=ix;
        blockIdx.x=t;blockIdx.y=0;
        cta(1024,[&]{indexer_top1_kernel(old.data()+4,x.data(),n,rows);});
    }
    unsigned chunks=(n+4095)/4096;
    std::vector<indexer_top1_pair> partials((size_t)rows*chunks+8,{-733.f,0x12345678u});
    for(unsigned t=0;t<rows;t++) for(unsigned c=0;c<chunks;c++) {
        blockIdx.x=t;blockIdx.y=c;
        cta(256,[&]{indexer_top1_chunks_kernel(partials.data()+4,x.data(),n,chunks);});
    }
    for(unsigned t=0;t<rows;t++) {
        blockIdx.x=t;blockIdx.y=0;
        cta(256,[&]{indexer_top1_finish_kernel(got.data()+4,partials.data()+4,chunks);});
        if(got[t+4]!=expected[t] || old[t+4]!=expected[t]) {
            fprintf(stderr,"n=%u rows=%u pattern=%u row=%u old=%u new=%u oracle=%u\n",n,rows,pattern,t,old[t+4],got[t+4],expected[t]);
            fail("top-1 differs");
        }
    }
    for(unsigned i=0;i<rows+8;i++) if(i<4 || i>=rows+4)
        need(got[i]==0xdeadbeefu && old[i]==0xdeadbeefu,"selected canary");
    for(size_t i=0;i<partials.size();i++) if(i<4 || i>=partials.size()-4)
        need(partials[i].value==-733.f && partials[i].index==0x12345678u,"partial canary");
    need(!memcmp(x.data(),saved.data(),x.size()*4),"scores mutated");cases++;
}
int main() {
    unsigned controls=0;
    auto shape=[&](unsigned n,unsigned r,uint64_t b){controls++;return indexer_top1_wide_shape(n,r,b);};
    for(unsigned n:{65536u,65537u,248320u,1048576u}) for(unsigned r:{1u,2u,7u}) {
        uint64_t bytes=(uint64_t)r*((n+4095u)/4096u)*8;
        need(shape(n,r,bytes),"eligible boundary");need(!shape(n,r,bytes-1),"scratch boundary");
    }
    for(unsigned n:{0u,1u,65535u,1048577u,0xffffffffu}) need(!shape(n,2,~0ull),"width fallback");
    for(unsigned r:{0u,8u,100u,0xffffffffu}) need(!shape(248320,r,~0ull),"row fallback");
    need(indexer_top1_ranges_overlap(16,8,20,8),"forward alias");
    need(indexer_top1_ranges_overlap(20,8,16,8),"reverse alias");
    need(indexer_top1_ranges_overlap(16,8,16,8),"same buffer alias");
    need(!indexer_top1_ranges_overlap(16,8,24,8),"adjacent ranges");
    for(unsigned p=0;p<8;p++) {trial(65537,2,p);trial(248320,2,p);}
    for(unsigned n:{1u,33u,4095u,4096u,4097u}) trial(n,1,3);
    trial(1048576,1,5);trial(248320,7,3);
    printf("Top-1 wide: %u exact kernel cases, %u shape controls,4 alias controls; oracle/old/new indices and guards PASS\n",cases,controls);
}
