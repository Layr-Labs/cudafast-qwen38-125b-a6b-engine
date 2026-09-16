/* Actual production bodies; fibers model CUDA threads, active-mask shuffles,
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

struct dim { uint32_t x=0; };
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
static void __syncthreads() {
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
static void atomicOr(uint32_t *p,uint32_t v) { *p|=v; }
static uint32_t __float_as_uint(float v) {
    uint32_t result;std::memcpy(&result,&v,4);return result;
}
using __half=uint16_t;
static __half __ushort_as_half(uint16_t h) { return h; }
static float __half2float(__half h) {
    _Float16 f;std::memcpy(&f,&h,2);return (float)f;
}
template<class T> static T __ldcs(const T *p) { return *p; }
static uint32_t __funnelshift_r(uint32_t lo,uint32_t hi,uint32_t s) {
    return s?(lo>>s)|(hi<<(32u-s)):lo;
}
static int __dp4a(int32_t a,int32_t b,int sum) {
    for(int i=0;i<4;i++)
        sum+=(int)(int8_t)((uint32_t)a>>(8*i))*(int)(int8_t)((uint32_t)b>>(8*i));
    return sum;
}
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#include "mtp_energy_bodies.inc"

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
static std::array<uint32_t,24> oracle_groups(const std::vector<int8_t> &q,
        const std::vector<float> &scales,uint32_t &invalid) {
    std::array<float,80> energy;
    std::array<uint32_t,80> order;
    for(unsigned g=0;g<80;g++) {
        int square=0;for(unsigned j=0;j<32;j++) square+=int(q[g*32+j])*int(q[g*32+j]);
        float e=scales[g]*(scales[g]*(float)square);
        if(!std::isfinite(e)||!std::isfinite(scales[g])) invalid|=1;
        energy[g]=std::isfinite(e)?e:0.f;
    }
    std::iota(order.begin(),order.end(),0u);
    std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) {
        return energy[a]>energy[b];
    });
    std::array<uint32_t,24> selected;
    std::copy_n(order.begin(),24,selected.begin());
    std::sort(selected.begin(),selected.end());
    return selected;
}
static std::array<uint32_t,24> select(const std::vector<int8_t> &q,
        const std::vector<float> &scales,uint32_t &invalid) {
    // Room for every group, so a broken publication barrier reports a canary
    // failure instead of corrupting the simulator's own stack.
    std::array<uint32_t,88> guarded;guarded.fill(0xfaceabcd);
    cta(128,[&]{mtp_native_energy_groups_kernel(guarded.data()+4,&invalid,q.data(),scales.data());});
    for(unsigned i=0;i<guarded.size();i++) if(i<4||i>=28)
        need(guarded[i]==0xfaceabcd,"selection canary");
    std::array<uint32_t,24> selected;
    std::copy_n(guarded.begin()+4,24,selected.begin());return selected;
}
static void group_tests() {
    for(unsigned pattern=0;pattern<80;pattern++) {
        std::vector<int8_t> q(2560);std::vector<float> scales(80);
        for(auto &v:q) v=(int8_t)random_word();
        for(auto &v:scales) v=std::ldexp(float(1+random_word()%17),int(random_word()%32)-16);
        if(pattern<6) {
            std::fill(q.begin(),q.end(),pattern==0?0:1);
            std::fill(scales.begin(),scales.end(),1.f);
        }
        if(pattern==2) for(unsigned g=0;g<80;g++) scales[g]=(float)(g+1);
        if(pattern==3) for(unsigned g=0;g<80;g++) scales[g]=(float)(80-g);
        if(pattern==4) for(unsigned g=0;g<80;g++) scales[g]=(float)(g%5);
        if(pattern==5) std::fill(q.begin(),q.end(),-128);
        if(pattern==6) scales[79]=INFINITY;
        if(pattern==7) scales[47]=NAN;
        if(pattern==8) scales[30]=1e30f;
        if(pattern==9) std::fill(scales.begin(),scales.end(),-0.f);
        auto before=q;auto sc_before=scales;
        uint32_t flag=0x10,expected_flag=flag;
        auto expected=oracle_groups(q,scales,expected_flag),got=select(q,scales,flag);
        need(got==expected,"stable top-energy groups");need(flag==expected_flag,"nonfinite flag");
        need(before==q&&!std::memcmp(sc_before.data(),scales.data(),320),"selection input mutation");
    }
    std::puts("80 production group-selection cases PASS");
}
static float scalar_dot(const unsigned char *row,const std::vector<int8_t> &q,
        const std::vector<float> &scales,const uint32_t *groups,unsigned count) {
    float sums[32]={};
    for(unsigned slot=0;slot<count;slot++) {
        unsigned g=groups?groups[slot]:slot;
        uint16_t h;std::memcpy(&h,row+g*34,2);
        int dot=0;
        for(unsigned j=0;j<32;j++) dot+=int((int8_t)row[g*34+2+j])*int(q[g*32+j]);
        sums[slot%32]=std::fma(__half2float(h)*scales[g],float(dot),sums[slot%32]);
    }
    for(unsigned step=16;step;step/=2)
        for(unsigned i=0;i<step;i++) sums[i]+=sums[i+step];
    return sums[0];
}
static void projection_test(unsigned offset,bool tail_energy) {
    constexpr unsigned vocab=73,prefix=61,tail=5,width=prefix+tail,row_bytes=2720;
    std::vector<unsigned char> map(offset+vocab*row_bytes);
    auto *w=map.data()+offset;
    std::vector<int8_t> q(2560);std::vector<float> scales(80);
    for(auto &v:q) v=(int8_t)(int(random_word()%255)-127);
    for(auto &v:scales) v=float(1+random_word()%32)*.00390625f;
    if(tail_energy) for(unsigned g=0;g<80;g++) scales[g]=g>=56?1.f:0.f;
    for(unsigned i=0;i<vocab;i++) for(unsigned g=0;g<80;g++) {
        uint16_t h=(uint16_t)(0x2800u+(random_word()%0x2000u));
        std::memcpy(w+i*row_bytes+g*34,&h,2);
        for(unsigned j=0;j<32;j++) w[i*row_bytes+g*34+2+j]=(unsigned char)random_word();
    }
    uint32_t flag=0;auto groups=select(q,scales,flag);need(flag==0,"finite metric");
    auto map_before=map;auto q_before=q;auto scales_before=scales;
    for(unsigned mode=0;mode<2;mode++) {
        const uint32_t *indices=mode?groups.data():nullptr;
        std::vector<float> scores(width+4,123.f);
        std::vector<uint64_t> keys(width+4,0xfaceabcd);
        for(blockIdx.x=0;blockIdx.x<(width+3)/4;blockIdx.x++) {
            cta(256,[&]{mtp_native_projection_kernel<true>(scores.data(),w,q.data(),scales.data(),width,nullptr,vocab,prefix,tail,nullptr,nullptr,indices);});
            cta(256,[&]{mtp_native_projection_kernel<true,true>(nullptr,w,q.data(),scales.data(),width,nullptr,vocab,prefix,tail,keys.data(),&flag,indices);});
        }
        for(unsigned r=0;r<width;r++) {
            unsigned id=r<prefix?r:vocab-tail+r-prefix;
            float want=scalar_dot(w+id*row_bytes,q,scales,indices,24);
            need(!std::memcmp(&want,&scores[r],4),"coarse dot scalar oracle");
            uint64_t key=(!id||r>=prefix)?UINT64_MAX-id:q8_top1_pack_key(want==0.f?0.f:want,id);
            need(keys[r]==key,"fused key oracle");
        }
        for(unsigned r=width;r<width+4;r++) need(scores[r]==123.f&&keys[r]==0xfaceabcd,"projection output canary");
    }
    // Noncontiguous refinement IDs: the energy map must not affect full dots.
    std::vector<uint32_t> ids={72,60,0,37,69,1,23,51,12,70,9,45,19};
    std::vector<float> scores(ids.size()+4,123.f);
    for(blockIdx.x=0;blockIdx.x<(ids.size()+3)/4;blockIdx.x++)
        cta(256,[&]{mtp_native_projection_kernel<false>(scores.data(),w,q.data(),scales.data(),(uint32_t)ids.size(),ids.data(),vocab,prefix,tail,nullptr,nullptr,groups.data());});
    for(unsigned r=0;r<ids.size();r++) {
        float want=scalar_dot(w+ids[r]*row_bytes,q,scales,nullptr,80);
        need(!std::memcmp(&want,&scores[r],4),"full refinement scalar oracle");
    }
    for(unsigned r=ids.size();r<scores.size();r++) need(scores[r]==123.f,"refinement canary");
    need(map==map_before&&q==q_before&&scales==scales_before,"projection input mutation");
}
static void shortlist_test(bool energy_helps) {
    constexpr unsigned prefix=2052,tail=3,vocab=prefix+tail,row_bytes=2720;
    constexpr unsigned winner=prefix-1;
    std::vector<unsigned char> w(vocab*row_bytes);
    std::vector<int8_t> q(2560,1);std::vector<float> scales(80,1.f);
    for(unsigned g=0;g<80;g++) {
        if(energy_helps&&g<56) std::fill_n(q.begin()+g*32,32,0);
        if(!energy_helps&&g>=56) scales[g]=2.f;
    }
    for(unsigned r=0;r<vocab;r++) for(unsigned g=0;g<80;g++) {
        uint16_t one=0x3c00;
        std::memcpy(w.data()+r*row_bytes+g*34,&one,2);
        unsigned value=energy_helps?(r==winner&&g>=56?2:0)
            :(r==winner?(g<24?127:0):(g>=56?1:0));
        std::memset(w.data()+r*row_bytes+g*34+2,value,32);
    }
    uint32_t flag=0;auto groups=select(q,scales,flag);
    float best=scalar_dot(w.data()+winner*row_bytes,q,scales,nullptr,80);
    for(unsigned r=0;r<vocab;r++) if(r!=winner)
        need(best>scalar_dot(w.data()+r*row_bytes,q,scales,nullptr,80),"constructed unique full winner");
    for(unsigned mode=0;mode<2;mode++) {
        std::vector<uint64_t> keys(vocab);
        for(blockIdx.x=0;blockIdx.x<(vocab+3)/4;blockIdx.x++)
            cta(256,[&]{mtp_native_projection_kernel<true,true>(nullptr,w.data(),q.data(),scales.data(),vocab,nullptr,vocab,prefix,tail,keys.data(),&flag,mode?groups.data():nullptr);});
        need(flag==0,"shortlist metric finite");
        std::sort(keys.begin(),keys.end(),std::greater<uint64_t>());
        bool included=false;
        for(unsigned i=0;i<MTP_NATIVE_CAP;i++)
            included|=UINT32_MAX-(uint32_t)keys[i]==winner;
        need(included==(energy_helps?(mode==1):(mode==0)),"shortlist retention direction");
    }
}
int main() {
    group_tests();
    for(unsigned offset:{0u,2u}) for(bool tail:{false,true}) projection_test(offset,tail);
    std::puts("4 production projection cases PASS: prefix/adaptive keys, sparse groups, exact refinement, alignment and canaries");
    shortlist_test(true);shortlist_test(false);
    std::puts("2 end-to-end shortlist cases PASS: recovery AND an explicit counterexample (heuristic is not universally better)");
}
