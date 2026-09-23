#include "kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
constexpr unsigned NK=16,NV=48,KD=NK*128,VD=NV*128,CD=2*KD+VD,TS=(KD+VD+2*NV+3)&~3u;
constexpr size_t CELLS=NV*128*128;
struct Buf {
    float *p; size_t n;
    Buf(size_t count):n(count){CK(cudaMalloc(&p,(n+8)*4)); clear();}
    ~Buf(){cudaFree(p);} float*d(){return p+4;}
    void clear(){std::vector<float> h(n+8,0); for(unsigned i=0;i<4;i++)h[i]=h[n+4+i]=12345.25f;CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));}
    void random(std::mt19937&r,float scale){std::vector<float> h(n);for(auto&v:h)v=(int(r()%2001)-1000)*scale;CK(cudaMemcpy(d(),h.data(),n*4,cudaMemcpyHostToDevice));}
    void guard(){std::vector<float> h(n+8);CK(cudaMemcpy(h.data(),p,h.size()*4,cudaMemcpyDeviceToHost));for(unsigned i=0;i<4;i++)if(h[i]!=12345.25f||h[n+4+i]!=12345.25f){fprintf(stderr,"canary changed\n");exit(2);}}
    void same(Buf&b,const char*label){std::vector<float>x(n),y(n);CK(cudaMemcpy(x.data(),d(),n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),b.d(),n*4,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),n*4)){fprintf(stderr,"mismatch %s\n",label);exit(3);}}
};
void copy(Buf&a,Buf&b){CK(cudaMemcpy(a.d(),b.d(),a.n*4,cudaMemcpyDeviceToDevice));}
void materialize(Buf&dst,Buf&cp,float*tape,unsigned rows,unsigned layout){
    materialize_kernel<<<dim3(NV,32),128>>>(nullptr,dst.d(),cp.d(),tape,nullptr,nullptr,nullptr,nullptr,NK,NV,0,layout,nullptr,rows);
    CK(cudaGetLastError());
}
template<unsigned Flush, bool Hybrid, bool FoldRow0>
void launch(float*out,float*state,float*cp,float*tape,float*qkv,float*gates,
            unsigned layout,uint32_t*control,cudaStream_t stream) {
#ifdef GDN_DEFERRED_PRODUCTION
    if constexpr (Flush==2 && !Hybrid && FoldRow0) {
        qwen4exp_gdn_deferred_kernel<true><<<dim3(NV,32),128,0,stream>>>(
            out,state,cp,tape,qkv,nullptr,nullptr,nullptr,nullptr,(float2*)gates,
            NK,NV,2,layout,control,0);
        return;
    }
#endif
    deferred_kernel<Flush,Hybrid,FoldRow0><<<dim3(NV,32),128,0,stream>>>(
        out,state,cp,tape,qkv,nullptr,nullptr,(float2*)gates,NK,NV,2,layout,control,0);
}

template<unsigned Flush, bool Hybrid=false, unsigned Streak=1, bool FoldRow0=false> void run(){
    std::mt19937 rng(78343+Flush); unsigned checked=0;
    for(unsigned layout=0;layout<2;layout++)for(unsigned pattern=0;pattern<5;pattern++){
        Buf qkv(2*CD),gates(4*NV),refcp(CELLS),refstate(CELLS),reftape(2*TS),refout(2*VD);
        Buf cp(CELLS),tape(2*(Flush+2)*TS),out(2*VD),finalstate(CELLS),row0(CELLS),refrow0(CELLS),unused(CELLS);
        uint32_t *control; CK(cudaMalloc(&control,4));
        refcp.random(rng,0.0001f);copy(cp,refcp);
        unsigned prefix=0,bank=0,phase=0,streak=Streak; bool eager=Hybrid;
        // Captured candidate reads the new control/data at stable addresses.
        cudaStream_t stream;cudaGraph_t graph[2];cudaGraphExec_t exec[2];CK(cudaStreamCreate(&stream));
        for(unsigned variant=0;variant<2;variant++){
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        launch<Flush,Hybrid,FoldRow0>(out.d(),variant?cp.d():unused.d(),variant?unused.d():cp.d(),tape.d(),qkv.d(),gates.d(),layout,control,stream);
        CK(cudaStreamEndCapture(stream,&graph[variant]));CK(cudaGraphInstantiate(&exec[variant],graph[variant],nullptr,nullptr,0));
        }
        for(unsigned round=0;round<32;round++){
            qkv.random(rng,pattern==4?1e-40f:0.0001f);
            std::vector<float> gp(4*NV);
            for(unsigned i=0;i<2*NV;i++){gp[2*i]=pattern==4?1.f:0.90f+float(rng()%100)*0.0009f;gp[2*i+1]=float(rng()%1000)*.001f;}
            CK(cudaMemcpy(gates.d(),gp.data(),gp.size()*4,cudaMemcpyHostToDevice));
            uint32_t word=prefix|(bank<<16u)|(unsigned(eager)<<17u);CK(cudaMemcpy(control,&word,4,cudaMemcpyHostToDevice));
            qwen4exp_gdn_replay_gates_kernel<<<dim3(NV,32),128>>>(refout.d(),refstate.d(),refcp.d(),reftape.d(),qkv.d(),nullptr,nullptr,(float2*)gates.d(),NK,NV,2,layout,nullptr,0);
            CK(cudaGetLastError());
            if(round&1)CK(cudaGraphLaunch(exec[phase],stream));
            else launch<Flush,Hybrid,FoldRow0>(out.d(),unused.d(),cp.d(),tape.d(),qkv.d(),gates.d(),layout,control,stream);
            CK(cudaDeviceSynchronize());
            const bool flush=prefix>=Flush || (eager && prefix);
            const unsigned folded=unsigned(FoldRow0 && flush);
            if(flush){prefix=0;bank^=1;}
            float*log=tape.d()+(size_t)bank*(Flush+2)*TS;
            materialize(finalstate,cp,log,prefix+2-folded,layout);
            materialize(row0,cp,log,prefix+1-folded,layout);
            materialize(refrow0,refcp,reftape.d(),1,layout);
            CK(cudaDeviceSynchronize());
            out.same(refout,"outputs");finalstate.same(refstate,"final state");row0.same(refrow0,"row-zero snapshot");
            for(Buf*b:{&qkv,&gates,&refcp,&refstate,&reftape,&refout,&cp,&tape,&out,&finalstate,&row0,&refrow0,&unused})b->guard();
            // Only eager mode may write the separate full-state destination.
            std::vector<float> sentinel(CELLS);CK(cudaMemcpy(sentinel.data(),unused.d(),CELLS*4,cudaMemcpyDeviceToHost));
            if(eager)unused.same(refstate,"eager final state");
            else if(std::any_of(sentinel.begin(),sentinel.end(),[](float x){return x!=0.f;}))exit(4);
            unsigned accepted=pattern==0?2:pattern==1?1:pattern==2?1+(round&1):1+(rng()&1);
            if(accepted==2)copy(refcp,refstate);else copy(refcp,refrow0);
            if(eager && accepted==2){std::swap(cp.p,unused.p);phase^=1;prefix=0;}else prefix+=accepted-folded;
            streak=accepted==2?std::min(Streak,streak+1):0;
            eager=Hybrid && streak==Streak;unused.clear();checked++;
        }
        for(unsigned variant=0;variant<2;variant++){CK(cudaGraphExecDestroy(exec[variant]));CK(cudaGraphDestroy(graph[variant]));}
        CK(cudaStreamDestroy(stream));CK(cudaFree(control));
    }
    printf("PASS Flush=%u Hybrid=%u Streak=%u FoldRow0=%u: %u forwards, eager/changed-input graphs, both head layouts, accepted/rejected sequences, final/row-zero state bitwise and guards\n",Flush,unsigned(Hybrid),Streak,unsigned(FoldRow0),checked);
}
template<unsigned Flush, bool Hybrid=false, unsigned Streak=1, bool FoldRow0=false> cudaGraphExec_t capture(
        Buf&a,Buf&b,Buf&log,Buf&qkv,Buf&gates,Buf&out,uint32_t*controls,
        const std::vector<unsigned>&accepted,cudaStream_t stream){
    constexpr unsigned layers=36;constexpr size_t log_stride=20*TS;
    std::vector<uint32_t> words(accepted.size());unsigned prefix=0,bank=0,phase=0,streak=Streak;
    cudaGraph_t graph;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(unsigned r=0;r<accepted.size();r++){
        if constexpr(Flush==0){
            if(r){if(accepted[r-1]==1)prefix=prefix==2?0:prefix+1;else{prefix=0;phase^=1;}}
            words[r]=prefix;
        }else words[r]=prefix|(bank<<16)|(unsigned(Hybrid && streak==Streak)<<17);
        for(unsigned l=0;l<layers;l++){
            float*cp=a.d()+l*CELLS;float*state=b.d()+l*CELLS;float*tape=log.d()+l*log_stride;
            if constexpr(Flush==0){
                if(phase)std::swap(cp,state);
                qwen4exp_gdn_replay_gates_kernel<<<dim3(NV,32),128,0,stream>>>(out.d()+l*2*VD,state,cp,tape,qkv.d()+l*2*CD,nullptr,nullptr,(float2*)(gates.d()+l*4*NV),NK,NV,2,0,controls+r,0);
            }else{
                if(phase)std::swap(cp,state);
                launch<Flush,Hybrid,FoldRow0>(out.d()+l*2*VD,state,cp,tape,qkv.d()+l*2*CD,gates.d()+l*4*NV,0,controls+r,stream);
            }
        }
        if constexpr(Flush!=0){
            bool eager=Hybrid && streak==Streak;
            const bool flush=prefix>=Flush || (eager && prefix);
            const unsigned folded=unsigned(FoldRow0 && flush);
            if(flush){prefix=0;bank^=1;}
            if(eager && accepted[r]==2){prefix=0;phase^=1;}else prefix+=accepted[r]-folded;
            streak=accepted[r]==2?std::min(Streak,streak+1):0;
        }
    }
    CK(cudaStreamEndCapture(stream,&graph));
    CK(cudaMemcpy(controls,words.data(),words.size()*4,cudaMemcpyHostToDevice));
    cudaGraphExec_t exec;CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));CK(cudaGraphDestroy(graph));return exec;
}
void benchmark(){
    constexpr unsigned layers=36,rounds=32,samples=9;
    std::mt19937 rng(93292);Buf a(layers*CELLS),b(layers*CELLS),log(layers*20*TS),qkv(layers*2*CD),gates(layers*4*NV),out(layers*2*VD);
    qkv.random(rng,.0001f);std::vector<float>gp(gates.n);for(size_t i=0;i<gp.size();i+=2){gp[i]=.97f;gp[i+1]=.5f;}CK(cudaMemcpy(gates.d(),gp.data(),gp.size()*4,cudaMemcpyHostToDevice));
    cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    for(unsigned pattern=0;pattern<7;pattern++){
        std::vector<unsigned>accepted(rounds);for(unsigned r=0;r<rounds;r++)accepted[r]=pattern==0?2:pattern==1?1:pattern==2?1+(r&1):pattern==4?1+((r/8)&1):1+(rng()%100<(pattern==5?25:pattern==6?90:63));
        uint32_t*control[8];for(auto&c:control)CK(cudaMalloc(&c,rounds*4));
        cudaGraphExec_t graph[8]={capture<0>(a,b,log,qkv,gates,out,control[0],accepted,stream),capture<2>(a,b,log,qkv,gates,out,control[1],accepted,stream),capture<4>(a,b,log,qkv,gates,out,control[2],accepted,stream),capture<8>(a,b,log,qkv,gates,out,control[3],accepted,stream),capture<2,true>(a,b,log,qkv,gates,out,control[4],accepted,stream),capture<2,true,2>(a,b,log,qkv,gates,out,control[5],accepted,stream),capture<2,false,1,true>(a,b,log,qkv,gates,out,control[6],accepted,stream),capture<2,true,2,true>(a,b,log,qkv,gates,out,control[7],accepted,stream)};
        std::vector<float>ms[8];
        for(unsigned sample=0;sample<samples+2;sample++)for(unsigned j=0;j<8;j++){
            unsigned mode=(j+sample)%8;
            CK(cudaMemsetAsync(a.d(),0,a.n*4,stream));CK(cudaMemsetAsync(b.d(),0,b.n*4,stream));CK(cudaMemsetAsync(log.d(),0,log.n*4,stream));
            CK(cudaEventRecord(begin,stream));CK(cudaGraphLaunch(graph[mode],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float value;CK(cudaEventElapsedTime(&value,begin,end));if(sample>=2)ms[mode].push_back(value/rounds);
        }
        for(unsigned mode=0;mode<8;mode++){std::sort(ms[mode].begin(),ms[mode].end());printf("BENCH pattern=%u mode=%u median_ms_per_36_layer_round=%.6f range=%.6f..%.6f\n",pattern,mode,ms[mode][samples/2],ms[mode].front(),ms[mode].back());CK(cudaGraphExecDestroy(graph[mode]));CK(cudaFree(control[mode]));}fflush(stdout);
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));CK(cudaStreamDestroy(stream));
}
#ifdef GDN_DEFERRED_PRODUCTION
__global__ void compute_pairs(float2*p,const float*a,const float*b,const float*c,const float*d) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<2*NV)p[i]=make_float2(expf(c[i%NV]*qwen4exp_gdn_softplus(a[i]+d[i%NV])),qwen4exp_gdn_sigmoid(b[i]));
}
void production_checks() {
    std::mt19937 rng(990931);unsigned checked=0;
    Buf qkv(2*CD),a(2*NV),b(2*NV),c(NV),d(NV),gates(4*NV);
    Buf cp0(CELLS),cp1(CELLS),log0(8*TS),log1(8*TS),out0(2*VD),out1(2*VD),state0(CELLS),state1(CELLS);
    for(unsigned layout=0;layout<2;layout++)for(unsigned bank=0;bank<2;bank++)
    for(unsigned prefix=0;prefix<4;prefix++)for(unsigned pattern=0;pattern<3;pattern++) {
        qkv.random(rng,pattern==2?1e-40f:.0001f);a.random(rng,pattern==1?.08f:.001f);
        b.random(rng,.001f);c.random(rng,.0001f);d.random(rng,.001f);
        cp0.random(rng,.0001f);copy(cp1,cp0);log0.random(rng,.0001f);copy(log1,log0);
        compute_pairs<<<1,128>>>((float2*)gates.d(),a.d(),b.d(),c.d(),d.d());
        unsigned word=prefix|(bank<<16);
        qwen4exp_gdn_deferred_kernel<false><<<dim3(NV,32),128>>>(out0.d(),state0.d(),cp0.d(),log0.d(),qkv.d(),a.d(),b.d(),c.d(),d.d(),nullptr,NK,NV,2,layout,nullptr,word);
        qwen4exp_gdn_deferred_kernel<true><<<dim3(NV,32),128>>>(out1.d(),state1.d(),cp1.d(),log1.d(),qkv.d(),a.d(),b.d(),c.d(),d.d(),(float2*)gates.d(),NK,NV,2,layout,nullptr,word);
        CK(cudaDeviceSynchronize());cp0.same(cp1,"raw checkpoint");log0.same(log1,"raw tape");out0.same(out1,"raw outputs");
        unsigned ob=bank^(prefix>=2),rows=prefix>=2?1:prefix+2;
        ds4_gpu_tensor st{state0.d(),CELLS*4},cp{cp0.d(),CELLS*4},tape{log0.d(),8*TS*4};
        if(!ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&cp,&tape,rows,ob,NK,NV,layout))exit(5);
        materialize(state1,cp1,log1.d()+(size_t)ob*4*TS,rows,layout);
        CK(cudaDeviceSynchronize());state0.same(state1,"production API final");
        if(!ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&cp,&tape,rows-1,ob,NK,NV,layout))exit(5);
        materialize(state1,cp1,log1.d()+(size_t)ob*4*TS,rows-1,layout);
        CK(cudaDeviceSynchronize());state0.same(state1,"production API row zero");
        for(Buf*x:{&cp0,&cp1,&log0,&log1,&out0,&out1,&state0,&state1})x->guard();
        if(!qwen4exp_deferred_range_safe(cp.ptr,cp.bytes,tape.ptr,tape.bytes,st.ptr,st.bytes) ||
           qwen4exp_deferred_range_safe((char*)cp.ptr+4,cp.bytes-4,tape.ptr,tape.bytes,st.ptr,st.bytes) ||
           qwen4exp_deferred_range_safe(cp.ptr,cp.bytes,(char*)cp.ptr+16,32,st.ptr,st.bytes) ||
           qwen4exp_deferred_range_safe(cp.ptr,cp.bytes,tape.ptr,tape.bytes,(char*)tape.ptr+16,16) ||
           qwen4exp_deferred_range_safe(cp.ptr,cp.bytes,tape.ptr,tape.bytes,(char*)cp.ptr+16,16) ||
           qwen4exp_deferred_range_safe(cp.ptr,cp.bytes,tape.ptr,tape.bytes,(void*)(UINTPTR_MAX-7u),16))exit(7);
        // Invalid requests must be rejected before any launch or write.
        ds4_gpu_tensor short_tape=tape;short_tape.bytes-=4;
        ds4_gpu_tensor misaligned=st;misaligned.ptr=(char*)st.ptr+4;/* Allocation has guard padding; retain capacity to test alignment. */
        ds4_gpu_tensor wrong_device=cp;wrong_device.device=1;
        if(ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&cp,&tape,4,ob,NK,NV,layout) ||
           ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&cp,&tape,0,2,NK,NV,layout) ||
           ds4_gpu_qwen4exp_gdn_deferred_materialize(&cp,&cp,&tape,0,ob,NK,NV,layout) ||
           ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&cp,&short_tape,0,ob,NK,NV,layout) ||
           ds4_gpu_qwen4exp_gdn_deferred_materialize(&misaligned,&cp,&tape,0,ob,NK,NV,layout) ||
           ds4_gpu_qwen4exp_gdn_deferred_materialize(&st,&wrong_device,&tape,0,ob,NK,NV,layout))exit(6);
        CK(cudaDeviceSynchronize());state0.same(state1,"invalid API wrote state");checked++;
    }
    printf("PASS production raw/precomputed gates and extracted materialization API: %u cases, all prefixes/banks/layouts, final/snapshot and invalid requests\n",checked);
}
#endif
int main(int argc,char**argv){if(argc==2&&!strcmp(argv[1],"--benchmark"))benchmark();else{run<2>();run<4>();run<8>();run<2,true>();run<2,true,2>();run<2,false,1,true>();run<2,true,2,true>();
#ifdef GDN_DEFERRED_PRODUCTION
production_checks();
#endif
}}
