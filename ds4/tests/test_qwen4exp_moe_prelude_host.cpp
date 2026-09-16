/* Run extracted production bodies. Fibers model CUDA lanes; every shuffle
 * waits for its entire warp and every CTA barrier waits for every thread.
 * The scheduler deliberately interleaves warps, exposing missing publication
 * barriers. No GPU timings or native math equivalence are inferred here. */
#include <ucontext.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

struct dim { uint32_t x = 0, y = 0, z = 0; };
static dim threadIdx, blockIdx, blockDim, gridDim;
struct lane_state {
    ucontext_t ctx{};
    std::vector<char> stack = std::vector<char>(32768);
    int state = 0;
};
static lane_state lanes[512];
static ucontext_t scheduler;
static int current, nth, arrivals[17];
static uint32_t lane_values[512];
static std::function<void()> body;

[[noreturn]] static void fail(const char *msg) {
    std::fprintf(stderr, "FAIL: %s (block %u lane %d)\n", msg, blockIdx.x, current);
    std::exit(1);
}
static void fiber_entry() {
    body();
    lanes[current].state = -1;
    swapcontext(&lanes[current].ctx, &scheduler);
    std::abort();
}
static void barrier(bool warp) {
    const int group = warp ? (current / 32) : 16;
    const int first = warp ? (current & ~31) : 0;
    const int count = warp ? 32 : nth;
    lanes[current].state = group + 1;
    if (++arrivals[group] == count) {
        for (int i = first; i < first + count; i++) {
            if (lanes[i].state != group + 1) fail("divergent barrier");
            lanes[i].state = 0;
        }
        arrivals[group] = 0;
    }
    swapcontext(&lanes[current].ctx, &scheduler);
}
static void __syncthreads() { barrier(false); }
static void __syncwarp() { barrier(true); }
template<class T> static T shuffle(T v, int source) {
    static_assert(sizeof(T) == 4, "32-bit CUDA lane value");
    std::memcpy(&lane_values[current], &v, 4);
    barrier(true);
    T out;
    std::memcpy(&out, &lane_values[(current & ~31) + source], 4);
    barrier(true);
    return out;
}
template<class T> static T __shfl_sync(uint32_t, T v, int source) {
    return shuffle(v, source);
}
template<class T> static T __shfl_down_sync(uint32_t, T v, uint32_t step) {
    const int lane = current & 31;
    return shuffle(v, lane + (int)step < 32 ? lane + step : lane);
}
template<class T> static T __shfl_up_sync(uint32_t, T v, uint32_t step) {
    const int lane = current & 31;
    return shuffle(v, lane >= (int)step ? lane - step : lane);
}
template<class T, bool Min> static T reduction(T v) {
    std::memcpy(&lane_values[current], &v, 4);
    barrier(true);
    T out = v;
    for (int i = current & ~31; i < (current & ~31) + 32; i++) {
        T a;
        std::memcpy(&a, &lane_values[i], 4);
        out = Min ? std::min(out, a) : std::max(out, a);
    }
    barrier(true);
    return out;
}
static uint32_t __reduce_max_sync(uint32_t, uint32_t v) {
    return reduction<uint32_t, false>(v);
}
static int32_t __reduce_min_sync(uint32_t, int32_t v) {
    return reduction<int32_t, true>(v);
}
static uint32_t __float_as_uint(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}
static float dev_qwen4exp_weight_value(uint32_t type, const char *p, uint32_t k) {
    if (type != 0) fail("host fixture only models the F32 shared gate");
    return ((const float *)p)[k];
}
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __CUDA_ARCH__ 1210
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_PDL_SYNC() ((void)0)
#define QWEN4EXP_MOE_SCAN_THREADS 512
#define QWEN4EXP_SHARED_GATE_STEPS 10u
static float ds4_qwen4exp_smem[512];
#include "moe_prelude_bodies.inc"

static uint32_t rng = 0x519987u;
static uint32_t random_word() { rng = rng * 1664525u + 1013904223u; return rng; }
static float random_float() { return (int32_t)(random_word() % 2049) / 128.0f - 8.0f; }

static void cta(uint32_t threads, std::function<void()> f) {
    body = std::move(f);
    nth = threads;
    blockDim.x = threads;
    std::fill(std::begin(arrivals), std::end(arrivals), 0);
    for (int i = 0; i < nth; i++) {
        auto &l = lanes[i];
        getcontext(&l.ctx);
        l.ctx.uc_stack.ss_sp = l.stack.data();
        l.ctx.uc_stack.ss_size = l.stack.size();
        l.ctx.uc_link = &scheduler;
        l.state = 0;
        makecontext(&l.ctx, fiber_entry, 0);
    }
    int cursor = 0;
    for (;;) {
        bool alive = false, ran = false;
        for (int step = 0; step < nth; step++) {
            const int i = (cursor + step) % nth;
            if (lanes[i].state != -1) alive = true;
            if (lanes[i].state != 0) continue;
            current = i;
            threadIdx.x = i;
            swapcontext(&scheduler, &lanes[i].ctx);
            cursor = (i + 73) % nth;
            ran = true;
            break;
        }
        if (!alive) break;
        if (!ran) fail("CTA/warp deadlock");
    }
}

template<class T> static void equal(const std::vector<T> &a,
                                  const std::vector<T> &b, const char *name) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size()*sizeof(T)))
        fail(name);
}
struct result {
    std::vector<int32_t> selected, count, offset, cursor, active, pairs, sum;
    std::vector<float> weights, mid, scales, gate;
    std::vector<int8_t> q;
    result(uint32_t rows, uint32_t ne, uint32_t k, uint32_t in, uint32_t mid_stride)
        : selected(rows*k+8, -99), count(ne+8, -99), offset(ne+8, -99),
          cursor(ne+8, -99), active(ne+9, -99), pairs(rows*k+8, -99),
          sum(rows*in/32+8, -99), weights(rows*k+8, -99.0f),
          mid(rows*mid_stride+8, -99.0f), scales(rows*in/32+8, -99.0f),
          gate(rows+8, -99.0f), q(rows*in+32, -99) {}
    void compare(const result &b) const {
#define EQ(n) equal(n, b.n, #n)
        EQ(selected); EQ(count); EQ(offset); EQ(cursor); EQ(active); EQ(pairs);
        EQ(sum); EQ(weights); EQ(mid); EQ(scales); EQ(gate); EQ(q);
#undef EQ
    }
};

template<bool Native> static void test(uint32_t rows, uint32_t ne, uint32_t k,
                                      uint32_t in, unsigned pattern) {
    const uint32_t md = 64u, stride = md * k + 13u;
    std::vector<float> logits(rows*ne+8, 91.0f), x(rows*in+8, 91.0f), router(in);
    for (uint32_t i=0; i<rows*ne; i++)
        logits[i] = pattern == 0 ? random_float() : pattern == 1
            ? (float)((i * 17u) % 9u) : (i & 1 ? -0.0f : 0.0f);
    for (uint32_t i=0; i<rows*in; i++) x[i] = random_float() * (pattern == 1 ? 1e-12f : .01f);
    for (auto &r : router) r = random_float();
    auto original_logits = logits, original_x = x;
    result a(rows, ne, k, in, stride), b = a;
    gridDim.x = rows;
    for (blockIdx.x=0; blockIdx.x<rows; blockIdx.x++)
        cta(32, [&] { qwen4exp_router_select_topk_kernel<Native>(a.selected.data(),
            a.weights.data(), logits.data(), ne, k, rows); });
    gridDim.x=1; blockIdx.x=0;
    cta(512, [&] { qwen4exp_moe_group_small_kernel(a.count.data(), a.offset.data(),
        a.cursor.data(), a.active.data(), a.pairs.data(), a.mid.data(),
        a.selected.data(), ne, rows*k, k, md, stride); });
    gridDim.x=in/32; gridDim.y=rows;
    for (blockIdx.y=0; blockIdx.y<rows; blockIdx.y++)
        for (blockIdx.x=0; blockIdx.x<in/32; blockIdx.x++)
            cta(32, [&] { qwen4exp_quantize_rows_kernel(a.q.data(), a.scales.data(),
                a.sum.data(), x.data(), in, in/32, in, 0, 1); });
    gridDim.x=rows; gridDim.y=1; blockIdx.y=0;
    for (blockIdx.x=0; blockIdx.x<rows; blockIdx.x++)
        cta(256, [&] { qwen4exp_shared_gate_kernel<0>(a.gate.data(),
            (const char *)router.data(), x.data(), 0, in, rows); });
    gridDim.x=1+rows+(rows*(in/32)+15)/16;
    for (blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x++)
        cta(512, [&] { qwen4exp_moe_prelude_kernel<Native>(b.selected.data(),
            b.weights.data(), logits.data(), b.count.data(), b.offset.data(),
            b.cursor.data(), b.active.data(), b.pairs.data(), b.mid.data(),
            b.q.data(), b.scales.data(), b.sum.data(), x.data(), b.gate.data(),
            router.data(), ne, k, rows, in, md, stride); });
    a.compare(b);
    equal(x, original_x, "input changed");
    equal(logits, original_logits, "logits changed");
    // Independent ordering and metadata oracle, including tie policy.
    for (uint32_t r=0; r<rows; r++) {
        std::vector<int> order(ne);
        for (uint32_t i=0; i<ne; i++) order[i]=i;
        std::stable_sort(order.begin(), order.end(), [&](int i,int j) {
            return logits[r*ne+i] > logits[r*ne+j];
        });
        for (uint32_t i=0; i<k; i++)
            if (b.selected[r*k+i] != order[i]) fail("independent top-k oracle");
    }
    int at=0, active=0;
    for (uint32_t e=0; e<ne; e++) {
        if (b.offset[e] != at) fail("independent offset oracle");
        int count=0;
        for (uint32_t p=0; p<rows*k; p++) if (b.selected[p]==(int)e) {
            if (b.pairs[at++]!=(int)p) fail("independent pair order");
            count++;
        }
        if (count && b.active[++active]!=(int)e) fail("independent active order");
        if (b.count[e]!=count || b.cursor[e]!=at) fail("independent counts");
    }
    if (b.active[0]!=active) fail("independent active count");
}

/* Also execute the production dispatch/validation wrapper, with only its
 * device calls stubbed. Unsupported and malformed inputs must not dispatch,
 * and a failed routed launch must not proceed to the shared computation. */
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device_id; };
struct ds4_gpu_qwen4exp_slab {
    const void *map; uint64_t map_size, offset, expert_bytes, row_bytes; uint32_t type;
};
enum { DS4_QWEN4EXP_TY_f32 = 0 };
struct qwen4exp_moe_prelude {
    const ds4_gpu_tensor *logits; ds4_gpu_tensor *shared_gate; const char *shared_router;
};
static int routed_calls, shared_calls, routed_ok = 1, shared_ok = 1;
static int cuda_current_tier() { return 0; }
static int ds4_tensor_device_idx(const ds4_gpu_tensor *t) { return t->device_id; }
static bool cuda_qwen4exp_moe_type_supported(uint32_t t) { return t == 0 || t == 8; }
static const char *cuda_resolve_weight_ptr(const void *map, uint64_t offset,
                                          uint64_t, int, const char *) {
    return (const char *)map + offset;
}
template<class... T> static int qwen4exp_routed_moe_cuda(T...) {
    routed_calls++; return routed_ok;
}
template<class... T> static int qwen4exp_shared_expert_cuda(T...) {
    shared_calls++; return shared_ok;
}
#include "moe_prelude_api.inc"

static void test_api() {
    const uint64_t need[] = {2*2560*4, 2*6400*4, 2*10*2560*4,
                            2*640*4, 2*4, 2*10*4, 2*10*4, 2*512*4, 2*2560*4};
    std::vector<char> allocations[9], map(2000000);
    ds4_gpu_tensor t[9];
    for (unsigned i=0; i<9; i++) {
        allocations[i].resize(need[i]*4);
        t[i] = {allocations[i].data(), need[i]*4, 0};
    }
    ds4_gpu_qwen4exp_slab routed[3]{}, shared[4];
    for (auto &sl : shared) sl = {map.data(), map.size(), 0, 0, 680, 8};
    shared[0].type = 0;
    auto call = [&](uint32_t rows=2, uint32_t in=2560, uint32_t used=10,
                    uint32_t stride=6400) {
        return ds4_gpu_qwen4exp_moe_prelude_tensor(&t[0],&t[1],&t[2],&t[3],&t[4],
            &t[5],&t[6],&t[7],&t[8],routed,shared,in,640,2560,512,used,rows,stride);
    };
    unsigned checks=0;
    auto expect = [&](int got, int want, int rc, int sc) {
        if (got!=want || routed_calls!=rc || shared_calls!=sc) fail("API dispatch");
        routed_calls=shared_calls=0; checks++;
    };
    for (uint32_t n=1; n<=7; n++) expect(call(n),1,1,1);
    expect(call(8),-1,0,0); expect(call(0),0,0,0);
    expect(call(2,32768),-1,0,0); expect(call(2,2559),-1,0,0);
    expect(call(2,2560,0),0,0,0); expect(call(2,2560,513),0,0,0);
    expect(call(2,2560,33,33*640),-1,0,0);
    expect(call(2,2560,10,6399),0,0,0);
    for (auto name : {"DS4_QWEN4EXP_NO_MOE_PRELUDE",
                     "DS4_QWEN4EXP_SERIAL_GROUP_SCAN",
                     "DS4_QWEN4EXP_NO_MOE_QUANT_REUSE",
                     "DS4_QWEN4EXP_GENERIC_EXPERTS"}) {
        setenv(name,"1",1); expect(call(),-1,0,0); unsetenv(name);
    }
    for (unsigned i=0; i<9; i++) {
        const auto saved=t[i];
        t[i].bytes=need[i]-1; expect(call(),0,0,0); t[i]=saved;
        t[i].device_id=1; expect(call(),0,0,0); t[i]=saved;
        t[i].ptr=nullptr; expect(call(),0,0,0); t[i]=saved;
        for (unsigned j=0; j<i; j++) {
            t[i].ptr=t[j].ptr; expect(call(),0,0,0); t[i]=saved;
        }
    }
    for (unsigned i=0; i<4; i++) {
        auto saved=shared[i];
        shared[i].offset=shared[i].map_size+1; expect(call(),0,0,0); shared[i]=saved;
        shared[i].map=nullptr; expect(call(),0,0,0); shared[i]=saved;
        if (i) { shared[i].type=999; expect(call(),0,0,0); shared[i]=saved; }
    }
    shared[0].type=8; expect(call(),-1,0,0); shared[0].type=0;
    routed_ok=0; expect(call(),0,1,0); routed_ok=1;
    shared_ok=0; expect(call(),0,1,1); shared_ok=1;
    std::printf("MoE prelude production API: %u dispatch/error cases PASS\n", checks);
}

int main() {
    test_api();
    unsigned cases = 0;
    for (uint32_t rows : {1u,2u,3u,4u,7u})
        for (unsigned pattern=0; pattern<3; pattern++) {
            const uint32_t ne = pattern == 0 ? 512 : pattern == 1 ? 33 : 31;
            const uint32_t k = pattern == 0 ? 10 : pattern == 1 ? 32 : 1;
            const uint32_t in = pattern == 0 ? 2560 : pattern == 1 ? 1056 : 32;
            test<true>(rows,ne,k,in,pattern);
            test<false>(rows,ne,k,in,pattern);
            cases += 2;
        }
    std::printf("MoE prelude production bodies: %u cases PASS; all bytes, canaries, "
                "top-k ties, metadata and input immutability\n", cases);
}
