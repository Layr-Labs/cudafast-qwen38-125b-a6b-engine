/* Startup-only A/B of the routed-down implementations on private synthetic
 * Q5_1 bytes. No model map, request, prompt or timing phase is consulted. */
__global__ static void qwen4exp_down_cache_pressure(volatile uint32_t *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * (size_t)blockDim.x)
        p[i] = p[i] * 1664525u + 1013904223u;
}

extern "C" const char *ds4_gpu_qwen4exp_down_tune(void) {
    static char report[256] = "dnTune[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_QWEN4EXP_NO_DOWN_AUTOTUNE") ||
        getenv("DS4_QWEN4EXP_NO_DOWN_PANEL_REUSE") ||
        getenv("DS4_QWEN4EXP_NO_DOWN_PANEL") ||
        getenv("DS4_QWEN4EXP_NO_DOWN_ASYNC")) {
        snprintf(report, sizeof report, "dnTune[disabled]"); return report;
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

    constexpr size_t weight_bytes = 20u * 1228800u;
    const size_t pressure_bytes = (size_t)l2 * 4u;
    char *weights = nullptr;
    unsigned char *host_weights = nullptr;
    uint32_t *pressure = nullptr;
    int32_t *metadata = nullptr;
    int8_t *mq = nullptr;
    float *ms = nullptr, *output = nullptr, *mid = nullptr;
    int32_t *msum = nullptr, *selected = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    const char *failure = "setup";
    bool success = false, prefer = true;
    double a[2][4][7] = {}, b[2][4][7] = {};
    do {
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMalloc((void **)&weights, weight_bytes) != cudaSuccess ||
            cudaMalloc((void **)&pressure, pressure_bytes) != cudaSuccess ||
            cudaMalloc((void **)&metadata, (512u * 3u + 513u + 20u) * 4u) != cudaSuccess ||
            cudaMalloc((void **)&mq, 20u * 640u) != cudaSuccess ||
            cudaMalloc((void **)&ms, 400u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&msum, 400u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&selected, 20u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&output, 5120u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&mid, 20u * 640u * 4u) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) break;
        host_weights = (unsigned char *)malloc(weight_bytes);
        if (!host_weights) break;
        uint32_t random = 0x82dff127u;
        for (size_t i = 0; i < weight_bytes; i++) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            host_weights[i] = (unsigned char)random;
        }
        for (size_t i = 0; i < weight_bytes; i += 24u) {
            const uint16_t scale = 0x1400u, minimum = (i & 24u) ? 0x1000u : 0x9000u;
            memcpy(host_weights + i, &scale, 2);
            memcpy(host_weights + i + 2u, &minimum, 2);
        }
        int8_t host_q[12800]; float host_s[400]; int32_t host_sum[400];
        for (unsigned g = 0; g < 400u; g++) {
            int32_t sum = 0;
            for (unsigned j = 0; j < 32u; j++) {
                const int v = (int)((g * 37u + j * 71u) % 255u) - 127;
                host_q[g * 32u + j] = (int8_t)v; sum += v;
            }
            host_sum[g] = sum;
            host_s[g] = (g & 1u) ? 0.00390625f : 0.001953125f;
        }
        if (cudaMemcpy(weights, host_weights, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(mq, host_q, sizeof host_q, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(ms, host_s, sizeof host_s, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(msum, host_sum, sizeof host_sum, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemsetAsync(pressure, 0, pressure_bytes, stream) != cudaSuccess) break;
        free(host_weights); host_weights = nullptr;

        auto routes = [&](unsigned pattern) -> bool {
            int32_t ids[20];
            for (unsigned i = 0; i < 10u; i++) {
                ids[i] = (int32_t)i;
                ids[10u+i] = pattern == 0 ? (int32_t)i + 10 :
                    pattern == 1 ? (int32_t)i :
                    pattern == 2 ? 9 - (int32_t)i :
                    i < 5u ? 4 - (int32_t)i : (int32_t)i + 5;
            }
            if (pattern == 4) { ids[0] = -1; ids[6] = 512; ids[14] = -7; }
            return cudaMemcpy(selected, ids, sizeof ids, cudaMemcpyHostToDevice) == cudaSuccess;
        };
        auto launch = [&](bool reuse) -> bool {
            int32_t *const counts = metadata, *const offsets = metadata + 512u;
            int32_t *const cursor = metadata + 1024u, *const active = metadata + 1536u;
            int32_t *const pairs = active + 513u;
            if (reuse) {
                qwen4exp_moe_group_plan_kernel<<<1,512,0,stream>>>(
                    counts, offsets, cursor, active, pairs, mid, selected, 512,20,10,640,6400);
                if (cudaGetLastError() != cudaSuccess) return false;
                qwen4exp_moe_down_reuse_kernel<QW_DOWN_REUSE_PANELS><<<320,256,23040,stream>>>(
                    output, weights, cursor, mq, ms, msum);
            } else {
                qwen4exp_moe_group_small_kernel<<<1,512,0,stream>>>(
                    counts, offsets, cursor, active, pairs, mid, selected, 512,20,10,640,6400);
                if (cudaGetLastError() != cudaSuccess) return false;
                qwen4exp_moe_down_q_kernel<2,DS4_QWEN4EXP_TY_q5_1,true,true,true><<<320,256,7680,stream>>>(
                    output, weights, selected, mq, ms, msum, 1228800,480,
                    DS4_QWEN4EXP_TY_q5_1,20,2560,2,512,10);
            }
            return cudaGetLastError() == cudaSuccess;
        };
        bool ok = true;
        float reference[5120], got[5120];
        failure = "verify";
        for (unsigned pattern = 0; pattern < 5u && ok; pattern++) {
            ok = routes(pattern) && launch(false) &&
                cudaMemcpy(reference, output, sizeof reference, cudaMemcpyDeviceToHost) == cudaSuccess &&
                launch(true) && cudaMemcpy(got, output, sizeof got, cudaMemcpyDeviceToHost) == cudaSuccess &&
                !memcmp(reference, got, sizeof reference);
            for (unsigned i = 0; i < 5120u && ok; i++) ok = isfinite(got[i]);
        }
        if (!ok) break;
        failure = "timing";
        for (unsigned regime = 0; regime < 2u && ok; regime++) {
            for (unsigned pattern = 0; pattern < 4u && ok; pattern++) {
                ok = routes(pattern) && launch(false) && launch(true) &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                for (unsigned pair = 0; pair < 7u && ok; pair++) {
                    for (unsigned leg = 0; leg < 2u && ok; leg++) {
                        const bool reuse = ((pair + leg) & 1u) != 0;
                        double total_us = 0;
                        for (unsigned repeat = 0; repeat < 8u && ok; repeat++) {
                            if (regime) {
                                qwen4exp_down_cache_pressure<<<1024,256,0,stream>>>(pressure, pressure_bytes/4u);
                                if (cudaGetLastError() != cudaSuccess) { ok = false; break; }
                            }
                            float ms_elapsed = 0;
                            ok = cudaEventRecord(begin, stream) == cudaSuccess && launch(reuse) &&
                                cudaEventRecord(end, stream) == cudaSuccess &&
                                cudaEventSynchronize(end) == cudaSuccess &&
                                cudaEventElapsedTime(&ms_elapsed, begin, end) == cudaSuccess &&
                                isfinite(ms_elapsed) && ms_elapsed > 0.0f && ms_elapsed < 1000.0f;
                            if (ok) total_us += (double)ms_elapsed * 1000.0;
                        }
                        (reuse ? b : a)[regime][pattern][pair] = total_us / 8.0;
                    }
                }
            }
        }
        if (!ok) break;
        success = true;
    } while (false);
    if (host_weights) free(host_weights);
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (mid) (void)cudaFree(mid);
    if (output) (void)cudaFree(output);
    if (selected) (void)cudaFree(selected);
    if (msum) (void)cudaFree(msum);
    if (ms) (void)cudaFree(ms);
    if (mq) (void)cudaFree(mq);
    if (metadata) (void)cudaFree(metadata);
    if (pressure) (void)cudaFree(pressure);
    if (weights) (void)cudaFree(weights);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report, sizeof report, "dnTune[failed=%s]", failure); return report;
    }
    double ratio[2][4] = {}, means_a[2] = {}, means_b[2] = {}, worst_sd = 0;
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
        const double sd = sqrt(variance/6.0);
        if (sd > worst_sd) worst_sd = sd;
        if (wins < least_wins) least_wins = wins;
        ratio[r][p] = sb/sa; means_a[r] += sa/28.0; means_b[r] += sb/28.0;
        prefer = prefer && sb < 0.98*sa && wins >= 6 && sd < 0.05;
    }
    qwen4exp_down_tuned_device = device;
    qwen4exp_down_prefer_reuse = prefer;
    snprintf(report,sizeof report,
        "dnTune[l2=%d a=%.4g,%.4g b=%.4g,%.4g r=%.3g,%.3g,%.3g,%.3g/%.3g,%.3g,%.3g,%.3g sd=%.3g win=%u use=%u]",
        l2,means_a[0],means_a[1],means_b[0],means_b[1],
        ratio[0][0],ratio[0][1],ratio[0][2],ratio[0][3],
        ratio[1][0],ratio[1][1],ratio[1][2],ratio[1][3],
        worst_sd,least_wins,(unsigned)prefer);
    return report;
}
