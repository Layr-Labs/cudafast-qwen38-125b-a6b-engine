"""Compile the production unpack/sort kernel and compare it on a CUDA GPU.

No model weights are needed. NVCC may point at a task-local CUDA toolkit.
"""
from pathlib import Path
import os, re, subprocess, tempfile

root = Path(__file__).resolve().parents[2]
source = (root / 'ds4/ds4_cuda_mtp_native.cuh').read_text()
start = source.index('__global__ __launch_bounds__(256) static void mtp_native_unpack_sort_ids(')
end = source.index('/* Moving key writes', start)
kernel = source[start:end]
cap = re.search(r'static constexpr uint32_t MTP_NATIVE_CAP = \d+u;', source)[0]
program = r'''
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
''' + cap + '\n' + kernel + r'''
__global__ void old_unpack(uint32_t *ids, const uint64_t *keys) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<MTP_NATIVE_CAP) ids[i]=UINT32_MAX-(uint32_t)keys[i];
}
int main() {
    constexpr unsigned n=MTP_NATIVE_CAP;
    uint64_t *keys; uint32_t *out,*tmp;
    OK(cudaMalloc(&keys,n*sizeof(*keys)));
    OK(cudaMalloc(&out,n*sizeof(*out)));
    OK(cudaMalloc(&tmp,n*sizeof(*tmp)));
    std::vector<uint64_t> packed(n);
    std::vector<uint32_t> expected(n),actual(n);
    std::mt19937 rng(20260916);
    unsigned cases=0;
    for(uint32_t vocab: {2049u,4096u,248320u,1048576u,2147483648u,UINT32_MAX}) {
        int bits=1; while(bits<32 && (1u<<bits)<vocab) ++bits;
        for(int pattern=0;pattern<6;++pattern) for(int repetition=0;repetition<16;++repetition) {
            for(unsigned i=0;i<n;++i) {
                uint32_t id = pattern==0 ? rng()%vocab : pattern==1 ? i :
                    pattern==2 ? vocab-1-i : pattern==3 ? 0 :
                    pattern==4 ? (i%2 ? vocab-1:0) : (1u<<(i%bits))%vocab;
                expected[i]=id;
                packed[i]=(uint64_t(rng())<<32)|(UINT32_MAX-id);
            }
            std::sort(expected.begin(),expected.end());
            OK(cudaMemcpy(keys,packed.data(),n*sizeof(*keys),cudaMemcpyHostToDevice));
            mtp_native_unpack_sort_ids<<<1,256>>>(out,keys,bits);
            OK(cudaGetLastError());
            OK(cudaMemcpy(actual.data(),out,n*sizeof(*out),cudaMemcpyDeviceToHost));
            if(actual!=expected) { fprintf(stderr,"mismatch vocab=%u pattern=%d\n",vocab,pattern); return 2; }
            ++cases;
        }
    }
    printf("PASS %u GPU cases (%u IDs each), original-ID order matches std::sort\n",cases,n);
    // Paired microbenchmark of the old unpack + device sort versus the actual
    // production replacement. This is not the Qwen engine benchmark.
    for(unsigned i=0;i<n;++i) packed[i]=uint64_t(UINT32_MAX-(rng()%248320u));
    OK(cudaMemcpy(keys,packed.data(),n*sizeof(*keys),cudaMemcpyHostToDevice));
    size_t bytes=0;
    OK(cub::DeviceRadixSort::SortKeys(nullptr,bytes,tmp,out,n,0,18));
    void *scratch; OK(cudaMalloc(&scratch,bytes));
    auto launch=[&](bool fused) {
        if(fused) mtp_native_unpack_sort_ids<<<1,256>>>(out,keys,18);
        else {
            old_unpack<<<(n+255)/256,256>>>(tmp,keys);
            OK(cub::DeviceRadixSort::SortKeys(scratch,bytes,tmp,out,n,0,18));
        }
        OK(cudaGetLastError());
    };
    for(int i=0;i<100;++i) { launch(false); launch(true); }
    OK(cudaDeviceSynchronize());
    cudaEvent_t begin,endEvent; OK(cudaEventCreate(&begin)); OK(cudaEventCreate(&endEvent));
    std::vector<float> oldTimes,newTimes;
    for(int pair=0;pair<10;++pair) for(int pass=0;pass<2;++pass) {
        bool fused=(pair+pass)%2;
        OK(cudaEventRecord(begin));
        for(int i=0;i<1000;++i) launch(fused);
        OK(cudaEventRecord(endEvent)); OK(cudaEventSynchronize(endEvent));
        float ms; OK(cudaEventElapsedTime(&ms,begin,endEvent));
        (fused ? newTimes:oldTimes).push_back(ms); // ms/1000 iterations == us/iteration
    }
    std::sort(oldTimes.begin(),oldTimes.end()); std::sort(newTimes.begin(),newTimes.end());
    printf("MICRO median old=%.4f us fused=%.4f us (10 alternating pairs, 1000 iterations each)\n",
        (oldTimes[4]+oldTimes[5])/2,(newTimes[4]+newTimes[5])/2);
    OK(cudaFree(scratch)); OK(cudaFree(keys)); OK(cudaFree(out)); OK(cudaFree(tmp));
    OK(cudaEventDestroy(begin)); OK(cudaEventDestroy(endEvent));
}
'''
with tempfile.TemporaryDirectory(prefix='mtp-sort-') as directory:
    path = Path(directory)
    (path / 'probe.cu').write_text(program)
    subprocess.run([os.environ.get('NVCC', 'nvcc'), '-std=c++17', '-O3',
                    '-arch=' + os.environ.get('CUDA_TEST_ARCH', 'sm_86'),
                    str(path / 'probe.cu'), '-o', str(path / 'probe')], check=True)
    subprocess.run([str(path / 'probe')], check=True)
