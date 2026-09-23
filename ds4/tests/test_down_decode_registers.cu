#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e)); std::exit(1); } } while(0)

template<class T> struct Device {
    T *p; size_t n;
    explicit Device(size_t count):n(count) { CUDA(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T))); }
    explicit Device(const std::vector<T>&v):Device(v.size()) { CUDA(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice)); }
    ~Device() { cudaFree(p); }
};
static uint32_t random_word() { static uint32_t x=0xdeface; x^=x<<13;x^=x>>17;x^=x<<5;return x; }
constexpr int GUARD=64;
constexpr int MODES=2;
static int timing_samples=9, timing_replays=4;

struct Case {
    int experts, groups, rows, type, total, weight_offset;
    std::vector<int> counts, offsets, pairs, active, sums;
    std::vector<char> weights;
    std::vector<int8_t> x;
    std::vector<float> scales;
    size_t rowbytes, expertbytes;
    Case(std::vector<int> c,int g,int r,int t,int offset_bytes=0):experts(c.size()),groups(g),rows(r),type(t),weight_offset(offset_bytes),counts(c) {
        total=std::accumulate(counts.begin(),counts.end(),0);
        offsets.resize(experts); active.push_back(0);
        int offset=0;
        for(int e=0;e<experts;e++) { offsets[e]=offset;offset+=counts[e];if(counts[e]) active.push_back(e); }
        active[0]=int(active.size())-1;
        pairs.resize(total);std::iota(pairs.begin(),pairs.end(),0);
        std::mt19937 rng(123);std::shuffle(pairs.begin(),pairs.end(),rng);
        rowbytes=size_t(groups)*(type==DS4_QWEN4EXP_TY_q5_1?24:34);
        expertbytes=rowbytes*rows;
        weights.resize(expertbytes*experts);
        for(size_t i=0;i<weights.size();i+=4) { uint32_t v=random_word();std::memcpy(weights.data()+i,&v,std::min(size_t(4),weights.size()-i)); }
        for(int e=0;e<experts;e++) for(int row=0;row<rows;row++) for(int k=0;k<groups;k++) {
            char *w=weights.data()+e*expertbytes+row*rowbytes+k*(type==DS4_QWEN4EXP_TY_q5_1?24:34);
            static const uint16_t finite_halves[]={0,0x8000,1,0x8001,0x03ff,0x0400,0x2400,0x3c00,0xbc00,0x7bff,0xfbff};
            uint16_t d=finite_halves[random_word()%11];std::memcpy(w,&d,2);
            if(type==DS4_QWEN4EXP_TY_q5_1) { uint16_t m=finite_halves[random_word()%11];std::memcpy(w+2,&m,2); }
        }
        x.resize(size_t(total)*groups*32);scales.resize(size_t(total)*groups);sums.resize(scales.size());
        for(size_t b=0;b<scales.size();b++) {
            scales[b]=float(1+random_word()%1000)*0.000001f;
            for(int k=0;k<32;k++) { x[b*32+k]=int8_t(int(random_word()%255)-127);sums[b]+=x[b*32+k]; }
        }
    }
};

template<int T,bool Wide6> static void dispatch(int mode,float *out,Case &c,Device<char>&w,Device<int8_t>&x,
    Device<float>&s,Device<int>&sum,Device<int>&p,Device<int>&counts,Device<int>&offsets,Device<int>&active,
    bool compact,int dq,cudaStream_t stream) {
    dim3 grid((c.rows+63)/64,compact?c.active[0]:c.experts);
#define ARGS out,w.p+c.weight_offset,x.p,s.p,sum.p,p.p,counts.p,offsets.p,compact?active.p:nullptr,c.expertbytes,c.rowbytes,c.type,c.groups,c.rows,dq
    if(mode==0) baseline::qwen4exp_moe_down_mma_kernel<T,Wide6><<<grid,128,0,stream>>>(ARGS);
    else candidate::qwen4exp_moe_down_mma_kernel<T,Wide6><<<grid,128,0,stream>>>(ARGS);
    CUDA(cudaGetLastError());
#undef ARGS
}

static void run(Case &c, bool compact, int dq, const char *label, bool timing=false, bool wide=true, bool generic=false) {
    std::vector<char> padded(c.weight_offset, char(0xa5));
    padded.insert(padded.end(),c.weights.begin(),c.weights.end());
    Device<char>w(padded);Device<int8_t>x(c.x);Device<float>s(c.scales);Device<int>sum(c.sums);
    Device<int>p(c.pairs),counts(c.counts),offsets(c.offsets),active(c.active);
    size_t n=size_t(c.total)*c.rows;
    Device<float>out(n+2*GUARD);std::vector<unsigned char>expected,bytes((n+2*GUARD)*sizeof(float));
    cudaStream_t stream;CUDA(cudaStreamCreate(&stream));
    auto launch=[&](int mode) {
#define DISPATCH(T,W) dispatch<T,W>(mode,out.p+GUARD,c,w,x,s,sum,p,counts,offsets,active,compact,dq,stream)
        if(generic) { DISPATCH(-1,false); }
        else if(c.type==DS4_QWEN4EXP_TY_q5_1) {
            if(wide) DISPATCH(DS4_QWEN4EXP_TY_q5_1,true); else DISPATCH(DS4_QWEN4EXP_TY_q5_1,false);
        } else {
            if(wide) DISPATCH(DS4_QWEN4EXP_TY_q8_0,true); else DISPATCH(DS4_QWEN4EXP_TY_q8_0,false);
        }
#undef DISPATCH
    };
    for(int mode=0;mode<MODES;mode++) {
        CUDA(cudaMemset(out.p,0xa5,bytes.size()));launch(mode);CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(bytes.data(),out.p,bytes.size(),cudaMemcpyDeviceToHost));
        for(int i=0;i<GUARD*4;i++) if(bytes[i]!=0xa5||bytes[bytes.size()-1-i]!=0xa5) {std::fprintf(stderr,"guard %s mode%d\n",label,mode);std::exit(1);}
        if(mode==0) expected=bytes;
        else if(bytes!=expected) {std::fprintf(stderr,"mismatch %s mode%d g%d\n",label,mode,c.groups);std::exit(1);}
    }
    // Independent scalar reference samples cover output mapping and integer decoding.
    for(int e=0;e<c.experts;e++) if(c.counts[e]) for(int sample=0;sample<3;sample++) {
        int pair=c.pairs[c.offsets[e]+sample%c.counts[e]],row=(sample*37)%c.rows;
        float acc=0;
        for(int g=0;g<c.groups;g++) {
            const char *block=c.weights.data()+e*c.expertbytes+row*c.rowbytes+g*(c.type==DS4_QWEN4EXP_TY_q5_1?24:34);
            __half dh;std::memcpy(&dh,block,2);float a=__half2float(dh),b=0;
            uint32_t high=0;
            if(c.type==DS4_QWEN4EXP_TY_q5_1) {__half mh;std::memcpy(&mh,block+2,2);b=__half2float(mh);std::memcpy(&high,block+4,4);}
            int dot=0;
            for(int k=0;k<32;k++) {
                int value=c.type==DS4_QWEN4EXP_TY_q8_0 ? int(int8_t(block[2+k])) : ((uint8_t(block[8+k%16])>>(k<16?0:4))&15)+int((high>>k)&1)*16;
                dot+=value*c.x[(size_t(pair)*c.groups+g)*32+k];
            }
            size_t at=size_t(pair)*c.groups+g;
            acc=std::fma(a*c.scales[at],float(dot),acc);
            acc=std::fma(b*c.scales[at],float(c.sums[at]),acc);
        }
        float actual;std::memcpy(&actual,expected.data()+(GUARD+size_t(pair)*c.rows+row)*4,4);
        if(std::memcmp(&acc,&actual,4)) {std::fprintf(stderr,"CPU mismatch %s\n",label);std::exit(1);}
    }
    if(timing) {
        cudaGraphExec_t graphs[MODES];std::vector<float>times[MODES];
        for(int mode=0;mode<MODES;mode++) {
            cudaGraph_t graph;CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(mode);
            CUDA(cudaStreamEndCapture(stream,&graph));CUDA(cudaGraphInstantiate(&graphs[mode],graph,0));CUDA(cudaGraphDestroy(graph));
            for(int i=0;i<3;i++) CUDA(cudaGraphLaunch(graphs[mode],stream));
        }
        CUDA(cudaStreamSynchronize(stream));cudaEvent_t start,stop;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&stop));
        for(int sample=0;sample<timing_samples;sample++) for(int order=0;order<MODES;order++) {
            int mode=(sample+order)%MODES;CUDA(cudaEventRecord(start,stream));
            for(int i=0;i<timing_replays;i++) CUDA(cudaGraphLaunch(graphs[mode],stream));
            CUDA(cudaEventRecord(stop,stream));CUDA(cudaEventSynchronize(stop));float ms;CUDA(cudaEventElapsedTime(&ms,start,stop));times[mode].push_back(ms/timing_replays);
        }
        std::vector<float> ratios;
        for(int i=0;i<timing_samples;i++) ratios.push_back(times[1][i]/times[0][i]);
        std::sort(ratios.begin(),ratios.end());
        std::printf("%s paired candidate/base median=%.6f\n",label,ratios[timing_samples/2]);
        for(int mode=0;mode<MODES;mode++) {std::sort(times[mode].begin(),times[mode].end());std::printf("%s mode=%d ms median=%.6f min=%.6f max=%.6f\n",label,mode,times[mode][timing_samples/2],times[mode].front(),times[mode].back());}
        // Replay the captured graphs with changed activations at the same addresses.
        for(size_t b=0;b<c.scales.size();b++) {
            c.scales[b]*=-0.5f;c.sums[b]=0;
            for(int k=0;k<32;k++) { c.x[b*32+k]=int8_t(-c.x[b*32+k]);c.sums[b]+=c.x[b*32+k]; }
        }
        CUDA(cudaMemcpyAsync(x.p,c.x.data(),c.x.size(),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(s.p,c.scales.data(),c.scales.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(sum.p,c.sums.data(),c.sums.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
        std::vector<unsigned char> replay;
        for(int mode=0;mode<MODES;mode++) {
            CUDA(cudaMemsetAsync(out.p,0xa5,bytes.size(),stream));CUDA(cudaGraphLaunch(graphs[mode],stream));CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaMemcpy(bytes.data(),out.p,bytes.size(),cudaMemcpyDeviceToHost));
            for(int i=0;i<GUARD*4;i++) if(bytes[i]!=0xa5||bytes[bytes.size()-1-i]!=0xa5) {std::fprintf(stderr,"replay guard %s\n",label);std::exit(1);}
            if(mode==0) { if(bytes==expected) {std::fprintf(stderr,"replay input did not change %s\n",label);std::exit(1);} replay=bytes; }
            else if(bytes!=replay) {std::fprintf(stderr,"replay mismatch %s\n",label);std::exit(1);}
            CUDA(cudaGraphExecDestroy(graphs[mode]));
        }
        CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(stop));
    }
    CUDA(cudaStreamDestroy(stream));
}

int main(int argc,char **argv) {
    if(argc==3) { timing_samples=std::atoi(argv[1]);timing_replays=std::atoi(argv[2]); }
    if(timing_samples<1||timing_samples>1000||timing_replays<1||timing_replays>1000) return 2;
    cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,0));std::printf("Device %s\n",prop.name);
    { cudaFuncAttributes a,b;
      CUDA(cudaFuncGetAttributes(&a,baseline::qwen4exp_moe_down_mma_kernel<DS4_QWEN4EXP_TY_q5_1,true>));
      CUDA(cudaFuncGetAttributes(&b,candidate::qwen4exp_moe_down_mma_kernel<DS4_QWEN4EXP_TY_q5_1,true>));
      std::printf("baseline reg%d shared%zu local%zu; candidate reg%d shared%zu local%zu\n",a.numRegs,a.sharedSizeBytes,a.localSizeBytes,b.numRegs,b.sharedSizeBytes,b.localSizeBytes);
    }
    int cases=0;
    for(int type:{DS4_QWEN4EXP_TY_q5_1,DS4_QWEN4EXP_TY_q8_0}) for(int groups:{1,3,4,5,20,33}) for(bool compact:{false,true}) for(int dq:{0,1}) for(int residue:{0,2,4,8}) for(bool wide:{false,true}) {
        Case c({0,1,7,8,9,15,16,17,31,32,33,47,48,63,64,65},groups,128,type,residue);
        run(c,compact,dq,"boundary",false,wide);cases++;
    }
    std::printf("PASS %d routing/group/format cases, %d variants, full bitwise comparison, guards and scalar samples\n",cases,MODES);
    for(int type:{DS4_QWEN4EXP_TY_q5_1,DS4_QWEN4EXP_TY_q8_0}) for(int residue:{0,2,4,8}) for(int dq:{0,1}) {
        Case c({0,1,7,8,9,31,32,33,65},20,128,type,residue);
        run(c,true,dq,"generic",false,false,true);
    }
    std::puts("PASS 16 generic-dispatch cases");
    for(int tokens:{64,256,1024}) for(int distribution=0;distribution<3;distribution++) {
        std::vector<int> counts(512);
        for(int p=0;p<tokens*10;p++) {
            int expert=distribution==0 ? p%512 : distribution==1 ? random_word()%512 : (p%5==0?random_word()%512:random_word()%32);
            counts[expert]++;
        }
        char label[80];std::snprintf(label,sizeof(label),"q5 tokens%d distribution%d",tokens,distribution);
        Case c(counts,20,2560,DS4_QWEN4EXP_TY_q5_1);run(c,true,1,label,true);
    }
    std::puts("PASS all performance cases and changed-input graph replays also match");
}
