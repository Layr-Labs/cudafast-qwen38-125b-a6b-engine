/* Actual production GDN bodies; fibers model CUDA threads, active-mask shuffles,
 * and block barriers. Independent scalar oracles check the output bits and
 * tie policy. This is not a GPU performance or native rounding test. */
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
static dim threadIdx, blockIdx;
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
template<class T> static T __shfl_down_sync(uint32_t mask,T v,int d) {
    int lane=current&31;
    return shuffle(mask,v,lane+d<32?lane+d:lane);
}
template<class T> static T __shfl_sync(uint32_t mask,T v,int lane) {
    return shuffle(mask,v,lane);
}
struct alignas(8) float2 { float x,y; };
struct alignas(16) float4 { float x,y,z,w; };
static float2 make_float2(float x,float y) { return {x,y}; }
static void __stcs(float4 *p,float4 v) { *p=v; }
#define __device__
#define __global__
#define __forceinline__ inline
#define QWEN4EXP_GDN_DIM 128u
#include "ds4_qwen4exp_gdn_replay.h"
#include "gdn_replay_bodies.inc"
static void cta(int nth,std::function<void()> f) {
    threads=nth;body=std::move(f);block_arrivals=0;
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
static uint32_t rng=0x296416u;
static uint32_t random_word() { rng=rng*1664525u+1013904223u;return rng; }

struct guarded {
    std::vector<float> a;
    explicit guarded(size_t n): a(n+8, -8191.25f) {}
    float *data() { return a.data()+4; }
    size_t size() const { return a.size()-8; }
    void check() const {
        for(size_t i=0;i<4;i++)
            need(a[i]==-8191.25f && a[a.size()-1-i]==-8191.25f,"buffer canary");
    }
};
static float random_float(float scale) {
    return scale * ((float)(int32_t)(random_word()>>8) / 8388608.f - 1.f);
}
static void same(float a,float b,const char *what) {
    if(std::memcmp(&a,&b,4)) {
        std::fprintf(stderr,"%s: %.9g != %.9g\n",what,a,b); fail(what);
    }
}
static void trial(unsigned nk,unsigned nv,unsigned layout,unsigned rounds,bool sample_rows) {
    const unsigned kd=nk*128, vd=nv*128, cd=2*kd+vd;
    const unsigned stride=(kd+vd+2*nv+3)&~3u;
    guarded old_state(vd*128), old_snap(vd*128), b0(vd*128), b1(vd*128);
    guarded tape(stride*DS4_QWEN4EXP_GDN_REPLAY_ROWS), materialized(vd*128);
    guarded out(2*vd), ref(2*vd), qkv(2*cd), alpha(2*nv), beta(2*nv);
    std::vector<float> decay(nv),bias(nv);
    for(unsigned i=0;i<vd*128;i++) old_state.data()[i]=b0.data()[i]=random_float(.03f);
    for(unsigned h=0;h<nv;h++) {
        decay[h]=-(.03f+float(h%5)*.04f); bias[h]=random_float(.2f);
    }
    guarded *live=&b0, *base=&b1;
    std::vector<unsigned> cells;
    for(unsigned h=0;h<nv;h++) for(unsigned y=0;y<32;y++)
        if(!sample_rows || y==0 || y==11 || y==31) cells.push_back(h*32+y);
    bool previous=false;
    unsigned prefix=0, pending=0, flushes=0;
    // Includes long rejection chains, full acceptance at capacity, and restart.
    const char *pattern="RRRAARRRRRARARAAA";
    for(unsigned r=0;r<rounds;r++) {
        auto p=ds4_qwen4exp_gdn_replay_plan(true,previous,prefix,2,2,1,7,pending,pending);
        need(!p.settle && p.active,"two-row replay continuation");
        if(p.swap) std::swap(live,base);
        prefix=p.prefix;
        need(prefix<=2,"bounded prefix");
        flushes += prefix==2;
        for(unsigned i=0;i<2*cd;i++) qkv.data()[i]=random_float(.08f);
        for(unsigned i=0;i<2*nv;i++) {
            alpha.data()[i]=random_float(2.f);beta.data()[i]=random_float(2.f);
        }
        if(r%2) std::reverse(cells.begin(),cells.end());
        for(unsigned c:cells) {
            blockIdx={c/32,c%32,0};
            cta(128,[&]{qwen4exp_gdn_recurrence_kernel<false>(
                ref.data(),old_state.data(),qkv.data(),alpha.data(),beta.data(),
                decay.data(),bias.data(),nullptr,old_snap.data(),
                nk,nv,1,2,layout,1,0,&pending);});
        }
        const auto tape_before=tape.a;
        for(unsigned c:cells) {
            blockIdx={c/32,c%32,0};
            cta(128,[&]{qwen4exp_gdn_replay_kernel(
                out.data(),live->data(),base->data(),tape.data(),qkv.data(),
                alpha.data(),beta.data(),decay.data(),bias.data(),nk,nv,2,
                layout,&prefix,0);});
        }
        if(prefix==2) need(tape.a==tape_before,"flush must not overwrite a tape reader");
        else for(unsigned i=0;i<prefix*stride;i++)
            same(tape.data()[i],tape_before[i+4],"unmodified replay prefix");
        for(unsigned c:cells) for(unsigned v=0;v<4;v++) {
            unsigned index=(c/32)*128+(c%32)*4+v;
            for(unsigned t=0;t<2;t++) same(out.data()[t*vd+index],ref.data()[t*vd+index],"output bits");
            for(unsigned k=0;k<128;k++)
                same(live->data()[index*128+k],old_state.data()[index*128+k],"final state bits");
        }
        const auto final_before=live->a;
        unsigned rows=prefix==2?0:prefix+1;
        for(unsigned c:cells) {
            blockIdx={c/32,c%32,0};
            cta(128,[&]{qwen4exp_gdn_replay_kernel(
                nullptr,materialized.data(),base->data(),tape.data(),nullptr,
                nullptr,nullptr,nullptr,nullptr,nk,nv,0,layout,nullptr,rows);});
        }
        for(unsigned c:cells) for(unsigned v=0;v<4;v++) {
            unsigned index=(c/32)*128+(c%32)*4+v;
            for(unsigned k=0;k<128;k++)
                same(materialized.data()[index*128+k],old_snap.data()[index*128+k],"row-zero snapshot bits");
        }
        need(live->a==final_before,"inspection preserves canonical state");
        for(auto g:{&old_state,&old_snap,&b0,&b1,&tape,&materialized,&out,&ref,&qkv,&alpha,&beta}) g->check();
        pending=pattern[r%16]=='R'?1:0;
        previous=true;
    }
    if(rounds>=4) need(flushes!=0,"flush exercised");
    std::printf("PASS kernel nk=%u nv=%u layout=%u rounds=%u value_blocks=%zu flushes=%u\n",
                nk,nv,layout,rounds,cells.size(),flushes);
}
int main() {
    trial(1,1,0,16,false);
    trial(2,4,0,7,false);
    trial(2,4,1,7,false);
    trial(16,48,1,4,true);
    std::puts("PASS actual recurrence output, final state, snapshot, bounded log, canaries");
}
