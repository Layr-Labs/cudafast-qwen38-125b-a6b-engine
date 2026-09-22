#include "kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

static constexpr size_t GUARD = 256;
struct Output {
    unsigned char *q, *s, *n;
    size_t qb, sb, nb;
    Output(size_t count, size_t groups): qb(count), sb(count / 32 * sizeof(float)), nb(groups * sizeof(float)) {
        CUDA(cudaMalloc(&q, qb + 2 * GUARD));
        CUDA(cudaMalloc(&s, sb + 2 * GUARD));
        CUDA(cudaMalloc(&n, nb + 2 * GUARD));
        clear();
    }
    void clear() {
        CUDA(cudaMemset(q, 0xa5, qb + 2 * GUARD));
        CUDA(cudaMemset(s, 0xa5, sb + 2 * GUARD));
        CUDA(cudaMemset(n, 0xa5, nb + 2 * GUARD));
    }
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> all;
        const unsigned char *pointers[] = {q, s, n};
        const size_t sizes[] = {qb, sb, nb};
        for (int k = 0; k < 3; ++k) {
            std::vector<unsigned char> bytes(sizes[k] + 2 * GUARD);
            CUDA(cudaMemcpy(bytes.data(), pointers[k], bytes.size(), cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < GUARD; ++j) {
                if (bytes[j] != 0xa5 || bytes[GUARD + sizes[k] + j] != 0xa5) {
                    std::fprintf(stderr, "output guard corrupted (%d)\n", k); std::exit(1);
                }
            }
            all.insert(all.end(), bytes.begin() + GUARD, bytes.end() - GUARD);
        }
        return all;
    }
    ~Output() { cudaFree(q); cudaFree(s); cudaFree(n); }
};

static void launch(int mode, Output &out, float *x, float *w, int streams, int rows,
                   float eps, float bias, int bf16, cudaStream_t stream) {
    auto q = reinterpret_cast<int8_t *>(out.q + GUARD);
    auto s = reinterpret_cast<float *>(out.s + GUARD);
    auto n = reinterpret_cast<float *>(out.n + GUARD);
    dim3 grid(streams, rows);
    if (mode == 0) baseline::qwen4exp_hc_norm_quant_kernel<BASELINE_STAGED><<<grid,256,0,stream>>>(q,s,n,x,w,streams*2560,2560,rows,eps,bias,bf16);
    if (mode == 1) candidate::qwen4exp_hc_norm_quant_kernel<1><<<grid,256,0,stream>>>(q,s,n,x,w,streams*2560,2560,rows,eps,bias,bf16);
    if (mode == 2) candidate::qwen4exp_hc_norm_quant_kernel<0><<<grid,256,0,stream>>>(q,s,n,x,w,streams*2560,2560,rows,eps,bias,bf16);
    CUDA(cudaGetLastError());
}

int main() {
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s; baseline=%s; PDL disabled in isolated test\n", prop.name, BASELINE_STAGED ? "previous staged" : "rolled");
    cudaFuncAttributes a[3];
    CUDA(cudaFuncGetAttributes(&a[0], baseline::qwen4exp_hc_norm_quant_kernel<BASELINE_STAGED>));
    CUDA(cudaFuncGetAttributes(&a[1], candidate::qwen4exp_hc_norm_quant_kernel<1>));
    CUDA(cudaFuncGetAttributes(&a[2], candidate::qwen4exp_hc_norm_quant_kernel<0>));
    for (int k=0;k<3;k++) std::printf("mode=%d registers=%d local_bytes=%zu shared_bytes=%zu\n", k,a[k].numRegs,a[k].localSizeBytes,a[k].sharedSizeBytes);
    cudaStream_t stream;
    CUDA(cudaStreamCreate(&stream));
    std::mt19937 rng(8675309);
    std::normal_distribution<float> normal(0, 1);
    int cases = 0;
    for (int streams : {1,4,8}) for (int rows : {1,2,4,7,16}) {
        size_t count = size_t(streams)*rows*2560;
        std::vector<float> hx(count), hw(streams*2560);
        float *dx, *dw;
        CUDA(cudaMalloc(&dx,count*sizeof(float)));
        CUDA(cudaMalloc(&dw,hw.size()*sizeof(float)));
        Output out(count, streams*rows);
        for (int pattern=0;pattern<6;pattern++) {
            for (size_t i=0;i<count;i++) {
                float value=normal(rng);
                if (pattern==1) value=i%2 ? 0.0f : -0.0f;
                if (pattern==2) value=std::ldexp(float(int(i%17)-8),-140);
                if (pattern==3) value=std::ldexp(value,int(i%25)-12);
                if (pattern==4) value=(i%2?1.0f:-1.0f)*(1.0f+float(i%8)/256.0f);
                if (pattern==5) value=std::ldexp(value,50);
                hx[i]=value;
            }
            for (float &v:hw) v=normal(rng);
            CUDA(cudaMemcpy(dx,hx.data(),count*sizeof(float),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(dw,hw.data(),hw.size()*sizeof(float),cudaMemcpyHostToDevice));
            for (int bf16 : {0,1}) for (float bias : {0.0f,1.0f}) {
                float eps = pattern%2 ? 1e-6f : 1e-5f;
                std::vector<unsigned char> expected;
                for (int mode=0;mode<3;mode++) {
                    out.clear();
                    launch(mode,out,dx,dw,streams,rows,eps,bias,bf16,stream);
                    CUDA(cudaStreamSynchronize(stream));
                    auto actual=out.read();
                    if (!mode) expected=actual;
                    else if (actual != expected) {
                        std::fprintf(stderr,"mismatch mode=%d streams=%d rows=%d pattern=%d bf16=%d bias=%g\n",mode,streams,rows,pattern,bf16,bias);
                        return 1;
                    }
                }
                ++cases;
            }
        }
        CUDA(cudaFree(dx)); CUDA(cudaFree(dw));
    }
    std::printf("PASS %d cases: bitwise Q8, quant scales, norm scales, output guards\n",cases);

    // Graph replay limits CPU launch overhead. Rotate order across 15 samples.
    for (int rows : {1,2,4,16}) {
        int streams=4;
        size_t count=size_t(streams)*rows*2560;
        std::vector<float> hx(count), hw(streams*2560);
        for(float &v:hx) v=normal(rng);
        for(float &v:hw) v=normal(rng);
        float *dx,*dw;
        CUDA(cudaMalloc(&dx,count*sizeof(float)));
        CUDA(cudaMalloc(&dw,hw.size()*sizeof(float)));
        CUDA(cudaMemcpy(dx,hx.data(),count*sizeof(float),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dw,hw.data(),hw.size()*sizeof(float),cudaMemcpyHostToDevice));
        Output out(count,streams*rows);
        cudaGraphExec_t graphs[3];
        constexpr int repeats=256;
        for (int mode=0;mode<3;mode++) {
            cudaGraph_t graph;
            CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            for (int j=0;j<repeats;j++) launch(mode,out,dx,dw,streams,rows,1e-5f,1.0f,1,stream);
            CUDA(cudaStreamEndCapture(stream,&graph));
            CUDA(cudaGraphInstantiate(&graphs[mode],graph,0));
            CUDA(cudaGraphDestroy(graph));
            for(int j=0;j<8;j++) CUDA(cudaGraphLaunch(graphs[mode],stream));
        }
        CUDA(cudaStreamSynchronize(stream));
        std::vector<float> times[3];
        cudaEvent_t start,stop;
        CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
        for(int sample=0;sample<15;sample++) for(int order=0;order<3;order++) {
            int mode=(sample+order)%3;
            CUDA(cudaEventRecord(start,stream));
            for(int j=0;j<16;j++) CUDA(cudaGraphLaunch(graphs[mode],stream));
            CUDA(cudaEventRecord(stop,stream));
            CUDA(cudaEventSynchronize(stop));
            float ms;
            CUDA(cudaEventElapsedTime(&ms,start,stop));
            times[mode].push_back(ms*1000.0f/(repeats*16));
        }
        for(int mode=0;mode<3;mode++) {
            std::sort(times[mode].begin(),times[mode].end());
            std::printf("rows=%d mode=%d graph_us median=%.4f min=%.4f max=%.4f\n",rows,mode,times[mode][7],times[mode].front(),times[mode].back());
            CUDA(cudaGraphExecDestroy(graphs[mode]));
        }
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
        CUDA(cudaFree(dx)); CUDA(cudaFree(dw));
    }
    CUDA(cudaStreamDestroy(stream));
}
