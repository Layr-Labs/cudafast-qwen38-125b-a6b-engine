/* Startup-only choice among exact prefill down tiles. All storage and data
 * below are private synthetic inputs, released before any request. No model,
 * request, prompt, score, clock phase or benchmark identity is inspected. */
#pragma once
static int qwen4exp_down_prefill_choice = 0;
static int qwen4exp_down_prefill_device = -1;

__global__ static void qwen4exp_down_probe_weights(uint32_t *w, size_t blocks) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < blocks; i += gridDim.x * (size_t)blockDim.x) {
        uint32_t x = (uint32_t)i * 1664525u + 1013904223u;
        w[i * 6u] = 0x1400u | ((i & 1u ? 0x1000u : 0x9000u) << 16);
        for (unsigned j = 1; j < 6u; j++) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            w[i * 6u + j] = x;
        }
    }
}

__global__ static void qwen4exp_down_probe_activation(
        int8_t *q, float *s, int32_t *sum, unsigned groups, unsigned pattern) {
    for (unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
         g < groups; g += gridDim.x * blockDim.x) {
        int total = 0;
        for (unsigned j = 0; j < 32u; j++) {
            const int v = pattern == 1u ? 0 :
                (int)((g * 37u + j * 71u) % 255u) - 127;
            q[g * 32u + j] = (int8_t)v; total += v;
        }
        sum[g] = total;
        s[g] = pattern == 2u ? 1.0e-20f : pattern == 3u ? 65536.0f :
            pattern == 4u ? 1.0e-40f : (g & 1u) ? 0.00390625f : 0.001953125f;
    }
}

extern "C" const char *ds4_gpu_qwen4exp_down_prefill_tune(void) {
    static char report[320] = "pfTune[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_QWEN4EXP_NO_DOWN_PREFILL_TUNE") ||
        getenv("DS4_QWEN4EXP_NO_DOWN_RAW_PIPE") ||
        getenv("DS4_QWEN4EXP_NO_DOWN_DQ") ||
        getenv("DS4_QWEN4EXP_NO_Q51_WIDE_LOAD")) {
        snprintf(report, sizeof report, "pfTune[disabled]"); return report;
    }
    constexpr unsigned experts = 64u, max_pairs = experts * 64u;
    constexpr size_t weight_bytes = experts * 1228800u;
    constexpr size_t output_bytes = ((size_t)max_pairs * 2560u + 64u) * 4u;
    int device = -1, l2 = -1;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (g_n_gpus != 1 || cudaGetDevice(&device) != cudaSuccess ||
        device != g_gpu[0].device_id ||
        cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone ||
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess ||
        l2 <= 0 || weight_bytes <= (size_t)l2 * 2u) return report;

    char *weights = nullptr;
    int32_t *metadata = nullptr, *msum = nullptr;
    int8_t *mq = nullptr;
    float *ms = nullptr, *output = nullptr;
    unsigned char *reference = nullptr, *got = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    double a[2][4][7] = {}, b[2][4][7] = {};
    const char *failure = "setup";
    bool success = false;
    do {
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMalloc((void **)&weights, weight_bytes) != cudaSuccess ||
            cudaMalloc((void **)&metadata, (experts * 3u + 1u + max_pairs) * 4u) != cudaSuccess ||
            cudaMalloc((void **)&mq, max_pairs * 640u) != cudaSuccess ||
            cudaMalloc((void **)&ms, max_pairs * 20u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&msum, max_pairs * 20u * 4u) != cudaSuccess ||
            cudaMalloc((void **)&output, output_bytes) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) break;
        reference = (unsigned char *)malloc(output_bytes);
        got = (unsigned char *)malloc(output_bytes);
        if (!reference || !got) break;
        qwen4exp_down_probe_weights<<<1024,256,0,stream>>>(
            (uint32_t *)(void *)weights, weight_bytes / 24u);
        if (cudaGetLastError() != cudaSuccess) break;
        auto routes = [&](unsigned pattern) -> bool {
            int32_t host[experts * 3u + 1u + max_pairs] = {};
            unsigned np = 0;
            const unsigned mixed[6] = {1u,3u,9u,17u,33u,63u};
            for (unsigned e = 0; e < experts; e++) {
                const unsigned count = pattern == 0u ? 1u : pattern == 1u ? 20u :
                    pattern == 2u ? 32u : pattern == 3u ? mixed[e % 6u] : 64u;
                host[e] = (int32_t)count; host[experts + e] = (int32_t)np;
                host[experts * 2u + 1u + e] = (int32_t)(experts - 1u - e);
                np += count;
            }
            host[experts * 2u] = (int32_t)experts;
            for (unsigned i = 0; i < np; i++)
                host[experts * 3u + 1u + i] = (int32_t)(np - 1u - i);
            return cudaMemcpy(metadata, host, sizeof host, cudaMemcpyHostToDevice) == cudaSuccess;
        };
        auto activation = [&](unsigned pattern) -> bool {
            qwen4exp_down_probe_activation<<<320,256,0,stream>>>(
                mq, ms, msum, max_pairs * 20u, pattern);
            return cudaGetLastError() == cudaSuccess;
        };
        auto launch = [&](int tile) -> bool {
            int32_t *const counts = metadata, *const offsets = metadata + experts;
            int32_t *const active = metadata + experts * 2u;
            int32_t *const pairs = metadata + experts * 3u + 1u;
#define QW_PREFILL_PROBE(K, M) K<DS4_QWEN4EXP_TY_q5_1,true><<< \
            dim3(2560u/(M),experts,1),QW_DOWN_MMA_THREADS,0,stream>>>( \
            output,weights,mq,ms,msum,pairs,counts,offsets,active, \
            1228800u,480u,DS4_QWEN4EXP_TY_q5_1,20u,2560u,1u)
            if (tile == 64) { QW_PREFILL_PROBE(qwen4exp_moe_down_raw64_kernel,64u); }
            else if (tile == 32) { QW_PREFILL_PROBE(qwen4exp_moe_down_raw_pipe_kernel,32u); }
            else { QW_PREFILL_PROBE(qwen4exp_moe_down_mma_kernel,64u); }
#undef QW_PREFILL_PROBE
            return cudaGetLastError() == cudaSuccess;
        };
        bool ok = true;
        failure = "verify";
        for (unsigned pattern = 0; pattern < 6u && ok; pattern++) {
            ok = routes(pattern) && activation(pattern) &&
                cudaMemsetAsync(output, 0x5a, output_bytes, stream) == cudaSuccess &&
                launch(0) && cudaMemcpy(reference, output, output_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
            for (unsigned variant = 0; variant < 2u; variant++) {
                const int tile = variant == 0u ? 64 : 32;
                if (!ok) break;
                ok = cudaMemsetAsync(output, 0x5a, output_bytes, stream) == cudaSuccess &&
                    launch(tile) && cudaMemcpy(got, output, output_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    !memcmp(reference, got, output_bytes);
            }
            for (size_t i = 0; i < output_bytes / 4u && ok; i++)
                ok = isfinite(((const float *)(const void *)reference)[i]);
        }
        if (!ok) break;
        failure = "timing";
        for (unsigned candidate = 0; candidate < 2u && ok; candidate++) {
            const int tile = candidate == 0u ? 64 : 32;
            for (unsigned pattern = 0; pattern < 4u && ok; pattern++) {
                ok = routes(pattern) && activation(pattern) && launch(0) && launch(tile) &&
                    cudaStreamSynchronize(stream) == cudaSuccess;
                for (unsigned pair = 0; pair < 7u && ok; pair++) {
                    for (unsigned leg = 0; leg < 2u && ok; leg++) {
                        const bool raw = ((pair + leg) & 1u) != 0u;
                        float elapsed = 0;
                        ok = cudaEventRecord(begin, stream) == cudaSuccess;
                        for (unsigned repeat = 0; repeat < 8u && ok; repeat++)
                            ok = launch(raw ? tile : 0);
                        ok = ok && cudaEventRecord(end, stream) == cudaSuccess &&
                            cudaEventSynchronize(end) == cudaSuccess &&
                            cudaEventElapsedTime(&elapsed, begin, end) == cudaSuccess &&
                            isfinite(elapsed) && elapsed > 0.0f && elapsed < 1000.0f;
                        if (ok) (raw ? b : a)[candidate][pattern][pair] = (double)elapsed * 125.0;
                    }
                }
            }
        }
        if (!ok) break;
        success = true;
    } while (false);
    if (got) free(got);
    if (reference) free(reference);
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (output) (void)cudaFree(output);
    if (msum) (void)cudaFree(msum);
    if (ms) (void)cudaFree(ms);
    if (mq) (void)cudaFree(mq);
    if (metadata) (void)cudaFree(metadata);
    if (weights) (void)cudaFree(weights);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report, sizeof report, "pfTune[failed=%s]", failure); return report;
    }
    double ratios[2][4] = {}, worst_upper[2] = {}, mean_a[2] = {}, mean_b[2] = {};
    unsigned least_wins[2] = {7u,7u};
    bool prefer[2] = {true,true};
    for (unsigned c = 0; c < 2u; c++) for (unsigned p = 0; p < 4u; p++) {
        double sa = 0, sb = 0, mean = 0, variance = 0;
        unsigned wins = 0;
        for (unsigned i = 0; i < 7u; i++) {
            sa += a[c][p][i]; sb += b[c][p][i]; mean += b[c][p][i] / a[c][p][i];
            wins += b[c][p][i] < a[c][p][i];
        }
        mean /= 7.0;
        for (unsigned i = 0; i < 7u; i++) {
            const double d = b[c][p][i] / a[c][p][i] - mean; variance += d * d;
        }
        const double upper = mean + 3.0 * sqrt(variance / 42.0);
        if (upper > worst_upper[c]) worst_upper[c] = upper;
        if (wins < least_wins[c]) least_wins[c] = wins;
        ratios[c][p] = sb / sa; mean_a[c] += sa / 28.0; mean_b[c] += sb / 28.0;
        prefer[c] = prefer[c] && sb < 0.98 * sa && wins >= 6u && upper < 0.98;
    }
    qwen4exp_down_prefill_device = device;
    qwen4exp_down_prefill_choice = prefer[0] ? 64 : 0;
    if (prefer[1] && (!prefer[0] || mean_b[1]/mean_a[1] < mean_b[0]/mean_a[0]))
        qwen4exp_down_prefill_choice = 32;
    snprintf(report, sizeof report,
        "pfTune[l2=%d a=%.4g,%.4g b=%.4g,%.4g r=%.3g,%.3g,%.3g,%.3g/%.3g,%.3g,%.3g,%.3g upper=%.3g,%.3g win=%u,%u use=%d]",
        l2,mean_a[0],mean_a[1],mean_b[0],mean_b[1],
        ratios[0][0],ratios[0][1],ratios[0][2],ratios[0][3],
        ratios[1][0],ratios[1][1],ratios[1][2],ratios[1][3],
        worst_upper[0],worst_upper[1],least_wins[0],least_wins[1],qwen4exp_down_prefill_choice);
    return report;
}
