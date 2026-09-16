/* Private startup comparison of two exact cooperative gate/up copy paths. */
#pragma once
static bool qwen4exp_gateup_copy_prefer = false;
static int qwen4exp_gateup_copy_device = -1;

__global__ static void qwen4exp_gateup_probe_weights(uint32_t *w, size_t blocks) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < blocks; i += gridDim.x * (size_t)blockDim.x) {
        uint32_t x = (uint32_t)i * 1664525u + 1013904223u;
        w[i * 36u] = 0x10001400u;
        for (unsigned j = 1; j < 36u; j++) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            w[i * 36u + j] = x;
        }
    }
}
__global__ static void qwen4exp_gateup_cache_pressure(volatile uint32_t *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * (size_t)blockDim.x)
        p[i] = p[i] * 1664525u + 1013904223u;
}

extern "C" const char *ds4_gpu_qwen4exp_gateup_copy_tune(void) {
    static char report[320] = "guCopy[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_QWEN4EXP_NO_GATEUP_COPY_TUNE") ||
        getenv("DS4_QWEN4EXP_NO_GATEUP_ASYNC_COPY")) {
        snprintf(report, sizeof report, "guCopy[disabled]"); return report;
    }
    int device = -1, l2 = -1;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (g_n_gpus != 1 || cudaGetDevice(&device) != cudaSuccess ||
        device != g_gpu[0].device_id ||
        cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone ||
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess ||
        l2 <= 0 || l2 > 32 * 1024 * 1024) return report;

    constexpr size_t weight_bytes = 2u * 20u * 921600u;
    constexpr unsigned output_floats = 2u * 6416u + 64u;
    const size_t pressure_bytes = (size_t)l2 * 4u;
    char *weights = nullptr;
    uint32_t *pressure = nullptr;
    int32_t *metadata = nullptr, *xsum = nullptr;
    int8_t *xq = nullptr;
    float *xs = nullptr, *router = nullptr, *output = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    double a[2][4][7] = {}, b[2][4][7] = {};
    const char *failure = "setup";
    bool success = false, prefer = true;
    do {
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMalloc((void **)&weights, weight_bytes) != cudaSuccess ||
            cudaMalloc((void **)&pressure, pressure_bytes) != cudaSuccess ||
            cudaMalloc((void **)&metadata, 1557u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&xq, 5120u) != cudaSuccess ||
            cudaMalloc((void **)&xs, 160u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&xsum, 160u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&router, 20u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&output, output_floats * 4u) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) break;
        qwen4exp_gateup_probe_weights<<<1024,256,0,stream>>>(
            (uint32_t *)(void *)weights, weight_bytes / 144u);
        if (cudaGetLastError() != cudaSuccess) break;
        int8_t hq[5120]; float hs[160], hr[20]; int32_t hsum[160];
        for (unsigned g = 0; g < 160u; g++) {
            int32_t sum = 0;
            for (unsigned j = 0; j < 32u; j++) {
                const int v = (int)((g * 37u + j * 71u) % 255u) - 127;
                hq[g * 32u + j] = (int8_t)v; sum += v;
            }
            hsum[g] = sum; hs[g] = (g & 1u) ? 0.00390625f : 0.001953125f;
        }
        for (unsigned i = 0; i < 20u; i++) hr[i] = 0.0625f + (float)i / 256.0f;
        if (cudaMemcpy(xq,hq,sizeof hq,cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(xs,hs,sizeof hs,cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(xsum,hsum,sizeof hsum,cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(router,hr,sizeof hr,cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemsetAsync(pressure,0,pressure_bytes,stream) != cudaSuccess) break;
        auto routes = [&](unsigned pattern) -> bool {
            int32_t ids[20], host[1557] = {};
            const unsigned np = pattern == 0u ? 10u : 20u;
            for (unsigned i = 0; i < 10u; i++) {
                ids[i] = (int32_t)i;
                ids[10u+i] = pattern == 1u ? (int32_t)i + 10 :
                    pattern == 2u ? 9 - (int32_t)i :
                    i < 5u ? 4 - (int32_t)i : (int32_t)i + 5;
            }
            if (pattern == 4u) { ids[0] = -1; ids[6] = 512; ids[14] = -7; }
            for (unsigned i = 0; i < np; i++)
                if (ids[i] >= 0 && ids[i] < 512) host[ids[i]]++;
            unsigned cursor = 0, nactive = 0;
            for (unsigned e = 0; e < 512u; e++) {
                host[512u+e] = (int32_t)cursor;
                if (host[e]) host[1025u+nactive++] = (int32_t)e;
                for (unsigned i = 0; i < np; i++)
                    if (ids[i] == (int32_t)e) host[1537u+cursor++] = (int32_t)i;
            }
            host[1024u] = (int32_t)nactive;
            for (unsigned i = 0; i < nactive/2u; i++) {
                const int32_t e = host[1025u+i];
                host[1025u+i] = host[1024u+nactive-i]; host[1024u+nactive-i] = e;
            }
            return cudaMemcpy(metadata,host,sizeof host,cudaMemcpyHostToDevice) == cudaSuccess;
        };
        auto launch = [&](bool async) -> bool {
#define QW_GATEUP_PROBE(K) K<2,DS4_QWEN4EXP_TY_q4_K,true,QW_GU_COOP_ROWS,true><<< \
            dim3(160u,20u,1u),256,0,stream>>>(output,weights,weights+weight_bytes/2u, \
            xq,xs,xsum,metadata+1537u,metadata,metadata+512u,metadata+1024u,router, \
            921600u,1440u,921600u,1440u,DS4_QWEN4EXP_TY_q4_K,DS4_QWEN4EXP_TY_q4_K, \
            80u,640u,6416u,10u)
            if (async) { QW_GATEUP_PROBE(qwen4exp_moe_gateup_async_kernel); }
            else { QW_GATEUP_PROBE(qwen4exp_moe_gateup_split_kernel); }
#undef QW_GATEUP_PROBE
            return cudaGetLastError() == cudaSuccess;
        };
        bool ok = true;
        float reference[output_floats], got[output_floats];
        failure = "verify";
        for (unsigned pattern = 0; pattern < 5u && ok; pattern++) {
            ok = routes(pattern) && cudaMemsetAsync(output,0x5a,sizeof reference,stream) == cudaSuccess &&
                launch(false) && cudaMemcpy(reference,output,sizeof reference,cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemsetAsync(output,0x5a,sizeof reference,stream) == cudaSuccess && launch(true) &&
                cudaMemcpy(got,output,sizeof got,cudaMemcpyDeviceToHost) == cudaSuccess &&
                !memcmp(reference,got,sizeof reference);
            for (unsigned i = 0; i < output_floats && ok; i++) ok = isfinite(reference[i]);
        }
        if (!ok) break;
        failure = "timing";
        for (unsigned regime = 0; regime < 2u && ok; regime++) {
            for (unsigned pattern = 0; pattern < 4u && ok; pattern++) {
                ok = routes(pattern) && launch(false) && launch(true) &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                for (unsigned pair = 0; pair < 7u && ok; pair++) {
                    for (unsigned leg = 0; leg < 2u && ok; leg++) {
                        const bool async = ((pair + leg) & 1u) != 0u;
                        double total_us = 0;
                        for (unsigned repeat = 0; repeat < 8u && ok; repeat++) {
                            if (regime) {
                                qwen4exp_gateup_cache_pressure<<<1024,256,0,stream>>>(pressure,pressure_bytes/4u);
                                if (cudaGetLastError() != cudaSuccess) { ok = false; break; }
                            }
                            float elapsed = 0;
                            ok = cudaEventRecord(begin,stream) == cudaSuccess && launch(async) &&
                                cudaEventRecord(end,stream) == cudaSuccess && cudaEventSynchronize(end) == cudaSuccess &&
                                cudaEventElapsedTime(&elapsed,begin,end) == cudaSuccess &&
                                isfinite(elapsed) && elapsed > 0.0f && elapsed < 1000.0f;
                            if (ok) total_us += (double)elapsed * 1000.0;
                        }
                        (async ? b : a)[regime][pattern][pair] = total_us / 8.0;
                    }
                }
            }
        }
        if (!ok) break;
        success = true;
    } while (false);
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (output) (void)cudaFree(output);
    if (router) (void)cudaFree(router);
    if (xsum) (void)cudaFree(xsum);
    if (xs) (void)cudaFree(xs);
    if (xq) (void)cudaFree(xq);
    if (metadata) (void)cudaFree(metadata);
    if (pressure) (void)cudaFree(pressure);
    if (weights) (void)cudaFree(weights);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report,sizeof report,"guCopy[failed=%s]",failure); return report;
    }
    double ratios[2][4] = {}, means_a[2] = {}, means_b[2] = {}, worst_upper = 0;
    unsigned least_wins = 7;
    for (unsigned r = 0; r < 2u; r++) for (unsigned p = 0; p < 4u; p++) {
        double sa = 0, sb = 0, mean = 0, variance = 0;
        unsigned wins = 0;
        for (unsigned i = 0; i < 7u; i++) {
            sa += a[r][p][i]; sb += b[r][p][i]; mean += b[r][p][i]/a[r][p][i];
            wins += b[r][p][i] < a[r][p][i];
        }
        mean /= 7.0;
        for (unsigned i = 0; i < 7u; i++) {
            const double d = b[r][p][i]/a[r][p][i] - mean; variance += d*d;
        }
        const double upper = mean + 3.0 * sqrt(variance / 42.0);
        if (upper > worst_upper) worst_upper = upper;
        if (wins < least_wins) least_wins = wins;
        ratios[r][p] = sb/sa; means_a[r] += sa/28.0; means_b[r] += sb/28.0;
        prefer = prefer && sb < 0.98*sa && wins >= 6u && upper < 0.98;
    }
    qwen4exp_gateup_copy_device = device;
    qwen4exp_gateup_copy_prefer = prefer;
    snprintf(report,sizeof report,
        "guCopy[l2=%d a=%.4g,%.4g b=%.4g,%.4g r=%.3g,%.3g,%.3g,%.3g/%.3g,%.3g,%.3g,%.3g upper=%.3g win=%u use=%u]",
        l2,means_a[0],means_a[1],means_b[0],means_b[1],
        ratios[0][0],ratios[0][1],ratios[0][2],ratios[0][3],
        ratios[1][0],ratios[1][1],ratios[1][2],ratios[1][3],
        worst_upper,least_wins,(unsigned)prefer);
    return report;
}
