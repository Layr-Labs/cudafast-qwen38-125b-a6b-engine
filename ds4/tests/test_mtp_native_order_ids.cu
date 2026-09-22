// Synthetic differential test of the extracted production kernel. No weights.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "production_select.cuh"

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
            cudaGetErrorString(e)); exit(1); } } while (0)

static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}

template<bool Ordered>
static void launch(uint64_t *keys, uint32_t n, uint32_t *ids, uint32_t cap,
                   uint32_t *hist, uint32_t *ctl, uint64_t *a, uint64_t *b,
                   uint32_t *sorted, uint32_t prefix, uint32_t tail,
                   uint32_t vocab, int blocks) {
    void *args[] = {&keys, &n, &ids, &cap, &hist, &ctl, &a, &b,
                    &sorted, &prefix, &tail, &vocab};
    CHECK(cudaLaunchCooperativeKernel((void *)mtp_native_select_kernel<Ordered>,
          dim3(blocks), dim3(256), args, 0, nullptr));
}

static void run_case(uint32_t n, uint32_t cap, uint32_t tail, uint32_t vocab,
                     int pattern, int grid_limit, std::mt19937 &rng, bool timing) {
    const uint32_t prefix = n - tail;
    std::vector<uint64_t> keys(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t id = i < prefix ? i : vocab - tail + i - prefix;
        uint32_t score = pattern == 0 ? 0x80000000u + (rng() % 0x7f000000u)
                       : pattern == 1 ? 0xbf800000u
                       : pattern == 2 ? 0xbf800000u + i % 7u
                       : pattern == 3 ? 0x80000000u + (n - i)
                                      : 0x80000000u + i;
        keys[i] = (!id || i >= prefix) ? UINT64_MAX - id
                    : ((uint64_t)score << 32) | (UINT32_MAX - id);
    }
    auto ref_keys = keys;
    std::sort(ref_keys.begin(), ref_keys.end(), std::greater<uint64_t>());
    std::vector<uint32_t> expected(cap);
    for (uint32_t i = 0; i < cap; ++i) expected[i] = UINT32_MAX - (uint32_t)ref_keys[i];
    std::sort(expected.begin(), expected.end());
    uint64_t *dk, *a, *b;
    uint32_t *ids, *sorted, *hist, *ctl;
    CHECK(cudaMalloc(&dk, n * 8ull));
    CHECK(cudaMalloc(&a, n * 8ull + 64));
    CHECK(cudaMalloc(&b, n * 8ull + 64));
    CHECK(cudaMalloc(&ids, cap * 4ull + 64));
    CHECK(cudaMalloc(&sorted, cap * 4ull + 64));
    CHECK(cudaMalloc(&hist, 256 * 4));
    CHECK(cudaMalloc(&ctl, 8));
    CHECK(cudaMemcpy(dk, keys.data(), n * 8ull, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(a, 0xa5, n * 8ull + 64));
    CHECK(cudaMemset(b, 0xa5, n * 8ull + 64));
    CHECK(cudaMemset(ids, 0xa5, cap * 4ull + 64));
    CHECK(cudaMemset(sorted, 0xa5, cap * 4ull + 64));
    int sm, b0, b1;
    CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b0,
        mtp_native_select_kernel<false>, 256, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1,
        mtp_native_select_kernel<true>, 256, 0));
    const int blocks = std::min({sm * b1, (int)n, grid_limit});
    require(blocks > 0, "positive cooperative occupancy");
    launch<true>(dk,n,ids,cap,hist,ctl,a,b,sorted,prefix,tail,vocab,blocks);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(cap);
    CHECK(cudaMemcpy(got.data(), sorted, cap * 4ull, cudaMemcpyDeviceToHost));
    if (got != expected) {
        fprintf(stderr,"n=%u cap=%u tail=%u vocab=%u pattern=%d grid=%d\n",
                n,cap,tail,vocab,pattern,blocks);
        require(false, "fused result equals CPU full-key sort then ID sort");
    }
    for (auto item : {std::make_pair((char *)a, n * 8ull),
                      std::make_pair((char *)b, n * 8ull),
                      std::make_pair((char *)ids, cap * 4ull),
                      std::make_pair((char *)sorted, cap * 4ull)}) {
        unsigned char guard[64];
        CHECK(cudaMemcpy(guard, item.first + item.second, 64, cudaMemcpyDeviceToHost));
        for (auto byte : guard) require(byte == 0xa5, "allocation end guard");
    }
    size_t bytes = 0;
    int bits = 1;
    while (bits < 32 && (1u << bits) < vocab) ++bits;
    CHECK(cub::DeviceRadixSort::SortKeys(nullptr,bytes,ids,sorted,cap,0,bits));
    void *tmp;
    CHECK(cudaMalloc(&tmp,bytes));
    launch<false>(dk,n,ids,cap,hist,ctl,a,b,sorted,prefix,tail,vocab,
                   std::min({sm * b0, (int)n, grid_limit}));
    CHECK(cub::DeviceRadixSort::SortKeys(tmp,bytes,ids,sorted,cap,0,bits));
    CHECK(cudaMemcpy(got.data(), sorted, cap * 4ull, cudaMemcpyDeviceToHost));
    require(got == expected, "existing select plus CUB equals CPU oracle");
    if (timing) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
        // Interleave modes across five trials; synthetic local timings only.
        for (int trial = 0; trial < 5; ++trial) for (int m = 0; m < 2; ++m) {
            int mode = m ^ (trial & 1);
            CHECK(cudaEventRecord(start));
            for (int j = 0; j < 30; ++j) {
                if (mode) launch<true>(dk,n,ids,cap,hist,ctl,a,b,sorted,
                                       prefix,tail,vocab,blocks);
                else {
                    launch<false>(dk,n,ids,cap,hist,ctl,a,b,sorted,prefix,tail,vocab,
                                   std::min(sm*b0,(int)n));
                    CHECK(cub::DeviceRadixSort::SortKeys(tmp,bytes,ids,sorted,cap,0,bits));
                }
            }
            CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
            float ms; CHECK(cudaEventElapsedTime(&ms,start,stop));
            printf("synthetic n=%u cap=%u trial=%d mode=%s us=%.3f\n",
                   n,cap,trial,mode?"fused":"separate",ms*1000/30);
        }
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    }
    CHECK(cudaFree(tmp)); CHECK(cudaFree(dk)); CHECK(cudaFree(a)); CHECK(cudaFree(b));
    CHECK(cudaFree(ids)); CHECK(cudaFree(sorted)); CHECK(cudaFree(hist)); CHECK(cudaFree(ctl));
}

int main() {
    int coop;
    CHECK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,0));
    require(coop, "cooperative launch supported");
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
    printf("device=%s, CUDA runtime=%d\n",prop.name,CUDART_VERSION);
    std::mt19937 rng(0x5eed);
    unsigned cases=0;
    for (uint32_t n : {2049u,4097u,16385u,98584u,248320u,1048576u}) {
        const uint32_t cap=n>16384u?16384u:2048u;
        for (int pattern=0;pattern<5;++pattern) for (int grid : {1,7,100000}) {
            run_case(n,cap,276u,n+100003u,pattern,grid,rng,false); ++cases;
        }
    }
    for (uint32_t tail : {1u,31u,32u,33u,2047u}) {
        run_case(4099u,2048u,tail,UINT32_MAX,2,7,rng,false); ++cases;
    }
    run_case(98584u,2048u,276u,248320u,0,100000,rng,true); ++cases;
    run_case(248320u,16384u,276u,248320u,0,100000,rng,true); ++cases;
    printf("PASS: %u differential cases, fallback comparisons and end guards\n",cases);
}
