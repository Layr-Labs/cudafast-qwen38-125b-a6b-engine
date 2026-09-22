#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "production.cuh"
#ifdef COMPARE_PREVIOUS
#include "previous.cuh"
#endif
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
__global__ void fill_weights(unsigned char *w, uint32_t n) {
    for(uint32_t b=blockIdx.x*blockDim.x+threadIdx.x;b<n;b+=gridDim.x*blockDim.x) {
        *(__half*)(w+(uint64_t)b*34)=__float2half((int(b%127)-63)*0.015625f);
        for(uint32_t j=0;j<32;j++) w[(uint64_t)b*34+2+j]=(b*173+j*29+(b>>8))&255;
    }
}
int main() {
    constexpr uint32_t vocab=248320;
    unsigned char *w; int8_t *x; float *s,*a,*b; uint32_t *ids;
    CHECK(cudaMalloc(&w,(size_t)vocab*80*34));
    CHECK(cudaMalloc(&x,5120)); CHECK(cudaMalloc(&s,160*sizeof(float)));
    CHECK(cudaMalloc(&a,(32768+2)*sizeof(float))); CHECK(cudaMalloc(&b,(32768+2)*sizeof(float)));
    CHECK(cudaMalloc(&ids,32768*sizeof(uint32_t)));
    fill_weights<<<512,256>>>(w,vocab*80);
    int cases=0;
    for(int seed=0;seed<7;seed++) {
        std::vector<uint32_t> permutation(vocab);
        std::iota(permutation.begin(),permutation.end(),0u);
        std::mt19937 generator(719u+seed);
        std::shuffle(permutation.begin(),permutation.end(),generator);
        std::vector<int8_t> hx(5120); std::vector<float> hs(160);
        for(int i=0;i<5120;i++) hx[i]=(i*71+(i>>5)*13+seed*53)%256-128;
        for(int i=0;i<160;i++) hs[i]=(i%19==0)?0.0f:(int((i*17+seed*7)%101)-50)*0.0009765625f;
        CHECK(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(s,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
        for(uint32_t cap: {1u,3u,7u,31u,2048u,16384u}) for(int pct: {0,25,75,100}) {
            uint32_t overlap=cap*pct/100;
            std::vector<uint32_t> hi(2*cap);
            // Both clustered and interleaved IDs, including vocabulary endpoints.
            for(uint32_t i=0;i<cap;i++) hi[i]=seed==1?i*7:i;
            for(uint32_t i=0;i<cap;i++) hi[cap+i]=i<overlap?hi[i]:
                (seed==1?i*7+1:vocab-cap+i);
            if(seed>=3) {
                for(uint32_t i=0;i<cap;i++) hi[i]=permutation[i];
                for(uint32_t i=0;i<cap;i++) hi[cap+i]=i<overlap?hi[i]:permutation[cap+i];
                std::sort(hi.begin(),hi.begin()+cap);
            }
            std::sort(hi.begin()+cap,hi.end());
            CHECK(cudaMemcpy(ids,hi.data(),hi.size()*4,cudaMemcpyHostToDevice));
            CHECK(cudaMemset(a,0xa5,(2*cap+2)*4)); CHECK(cudaMemset(b,0xa5,(2*cap+2)*4));
            auto baseline=[&]() {
                for(int r=0;r<2;r++) mtp_native_projection_kernel<false><<<(cap+3)/4,256>>>(
                    a+1+r*cap,w,x+r*2560,s+r*80,cap,ids+r*cap,vocab,0,0);
            };
            auto paired=[&]() { mtp_native_projection2_refine_kernel<<<(2*cap+3)/4,256>>>(
                b+1,w,x,s,cap,ids,vocab); };
            baseline(); paired(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            std::vector<uint32_t> ha(2*cap+2),hb(2*cap+2);
            CHECK(cudaMemcpy(ha.data(),a,ha.size()*4,cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hb.data(),b,hb.size()*4,cudaMemcpyDeviceToHost));
            if(ha!=hb || hb.front()!=0xa5a5a5a5 || hb.back()!=0xa5a5a5a5) {
                for(size_t i=0;i<ha.size();i++) if(ha[i]!=hb[i]) {
                    fprintf(stderr,"Mismatch seed=%d cap=%u overlap=%d at=%zu %08x != %08x\n",
                        seed,cap,pct,i,ha[i],hb[i]); break;
                }
                return 1;
            }
            ++cases;
            if((seed==0 || seed==3) && cap==16384) {
                cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
#ifdef COMPARE_PREVIOUS
                constexpr int variants=3;
                auto previous=[&]() { previous_refine_kernel<<<(2*cap+3)/4,256>>>(
                    b+1,w,x,s,cap,ids,vocab); };
                previous(); CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(hb.data(),b,hb.size()*4,cudaMemcpyDeviceToHost));
                if(ha!=hb) { fprintf(stderr,"Previous candidate mismatch\n"); return 1; }
#else
                constexpr int variants=2;
#endif
                float times[variants]={};
                for(int repeat=0;repeat<6;repeat++) for(int turn=0;turn<variants;turn++) {
                    int variant=(repeat+turn)%variants;
                    CHECK(cudaEventRecord(start));
                    for(int iter=0;iter<20;iter++) {
                        if(variant==0) baseline();
                        else if(variant==1) paired();
#ifdef COMPARE_PREVIOUS
                        else previous();
#endif
                    }
                    CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
                    float ms; CHECK(cudaEventElapsedTime(&ms,start,stop)); times[variant]+=ms/120;
                }
                printf("Synthetic only seed=%d cap=%u overlap=%d%% baseline=%.3fms paired=%.3fms ratio=%.3f",
                    seed,cap,pct,times[0],times[1],times[0]/times[1]);
#ifdef COMPARE_PREVIOUS
                printf(" previous=%.3fms previous/new=%.3f",times[2],times[2]/times[1]);
#endif
                printf("\n");
                CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
            }
        }
    }
    printf("PASS %d bitwise comparisons including output guards\n",cases);
    CHECK(cudaFree(w)); CHECK(cudaFree(x)); CHECK(cudaFree(s)); CHECK(cudaFree(a));
    CHECK(cudaFree(b)); CHECK(cudaFree(ids));
}
