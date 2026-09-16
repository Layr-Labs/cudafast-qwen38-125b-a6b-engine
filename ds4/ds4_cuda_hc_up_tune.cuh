/* Startup-only exact-HC-up comparison. Private synthetic Q8 bytes and
 * activations, never model weights or request data. No inference allocations
 * or timing. A failed check or ambiguous measurement keeps the old kernel. */
static const char *hc_up_exact_tune(void) {
    static char report[192] = "hcUp[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_Q8_NO_HC_UP_MMA") || getenv("DS4_Q8_NO_HC_UP_TUNE") ||
        getenv("DS4_QWEN4EXP_NO_ROW_TILE") || getenv("DS4_Q8_NO_STREAM_LOADS") ||
        getenv("DS4_Q8_NO_HC_WARP_PAIR")) {
        snprintf(report, sizeof report, "hcUp[disabled]"); return report;
    }
    int device = -1, major = 0, l2 = 0;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (g_n_gpus != 1 || cudaGetDevice(&device) != cudaSuccess ||
        device != g_gpu[0].device_id ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major < 8 ||
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess ||
        l2 <= 0 || l2 > 128 * 1024 * 1024 ||
        cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError(); return report;
    }
    constexpr unsigned channels = 10240u;
    constexpr size_t panel_bytes = channels * 340u;
    const unsigned panels = (unsigned)((2ull * (unsigned)l2 + panel_bytes - 1u) / panel_bytes) + 1u;
    const size_t weight_bytes = (size_t)panels * panel_bytes;
    unsigned char *weights = nullptr, *host_weights = nullptr;
    int8_t *xq = nullptr;
    float *xs = nullptr, *aout = nullptr, *bout = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    const char *failure = "setup";
    bool success = false;
    double a[4][7] = {}, b[4][7] = {};
    do {
        host_weights = (unsigned char *)malloc(weight_bytes);
        if (!host_weights || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMalloc((void **)&weights, weight_bytes) != cudaSuccess ||
            cudaMalloc((void **)&xq, 640u) != cudaSuccess ||
            cudaMalloc((void **)&xs, 80u) != cudaSuccess ||
            cudaMalloc((void **)&aout, 2u * channels * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void **)&bout, 2u * channels * sizeof(float)) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) break;
        uint32_t state = 0x7b9125e3u;
        auto random_word = [&]() -> uint32_t {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
        };
        for (size_t block = 0; block < weight_bytes / 34u; block++) {
            unsigned char *p = host_weights + block * 34u;
            uint16_t scale = (uint16_t)(0x1000u | (random_word() & 0x83ffu));
            memcpy(p, &scale, 2u);
            for (unsigned k = 2; k < 34u; k++) p[k] = (unsigned char)random_word();
        }
        if (cudaMemcpy(weights, host_weights, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess) break;
        auto launch = [&](bool mma, unsigned rows, unsigned panel) -> bool {
            const unsigned char *w = weights + (size_t)panel * panel_bytes;
            if (mma) {
                QWEN4EXP_LAUNCH_PDL(matmul_q8_hc_up_exact_mma_kernel,
                        channels / 32u, 128, 0, stream, bout, w, xq, xs, rows, channels);
            } else {
                QWEN4EXP_LAUNCH_PDL((matmul_q8_hc_warp_pair_kernel<2>),
                        channels / 4u, 128, 0, stream, aout, w, xq, xs, (uint64_t)channels, rows);
            }
            return cudaGetLastError() == cudaSuccess;
        };
        int8_t hx[640]; float hs[20], old_values[2u * channels], new_values[2u * channels];
        bool ok = true;
        failure = "verify";
        for (unsigned pattern = 0; pattern < 6u && ok; pattern++) {
            for (unsigned i = 0; i < 640u; i++) hx[i] = pattern == 0 ? 0 :
                pattern == 1 ? -128 : pattern == 2 ? 127 : (int8_t)random_word();
            for (unsigned i = 0; i < 20u; i++) {
                const float magnitude = pattern == 3 ? 1.0e-38f :
                                        pattern == 4 ? 1.0e10f : 0.003f;
                hs[i] = (i & 1u) ? -magnitude : magnitude;
            }
            if (cudaMemcpy(xq, hx, sizeof hx, cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(xs, hs, sizeof hs, cudaMemcpyHostToDevice) != cudaSuccess) { ok = false; break; }
            for (unsigned rows = 1; rows <= 2u && ok; rows++) {
                const size_t bytes = (size_t)rows * channels * sizeof(float);
                if (!launch(false, rows, pattern % panels) || !launch(true, rows, pattern % panels) ||
                    cudaStreamSynchronize(stream) != cudaSuccess ||
                    cudaMemcpy(old_values, aout, bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    cudaMemcpy(new_values, bout, bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    memcmp(old_values, new_values, bytes)) ok = false;
            }
        }
        if (!ok) break;
        failure = "warmup";
        for (unsigned i = 0; i < 4u && ok; i++)
            for (unsigned rows = 1; rows <= 2u && ok; rows++)
                ok = launch(false, rows, i % panels) && launch(true, rows, i % panels);
        if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) break;
        failure = "timing";
        /* Modes: rows 1/2, hot single matrix / a rotation exceeding 2*L2.
         * Each leg covers complete rotations; alternate AB and BA. */
        for (unsigned mode = 0; mode < 4u && ok; mode++) {
            const unsigned rows = (mode & 1u) + 1u;
            const unsigned repetitions = mode < 2u ? 32u : panels * 2u;
            for (unsigned pair = 0; pair < 7u && ok; pair++) {
                for (unsigned leg = 0; leg < 2u && ok; leg++) {
                    const bool mma = ((pair + leg) & 1u) != 0;
                    if (cudaEventRecord(begin, stream) != cudaSuccess) { ok = false; break; }
                    for (unsigned i = 0; i < repetitions && ok; i++)
                        ok = launch(mma, rows, mode < 2u ? 0u : i % panels);
                    if (!ok || cudaEventRecord(end, stream) != cudaSuccess ||
                        cudaEventSynchronize(end) != cudaSuccess) { ok = false; break; }
                    float ms = 0.0f;
                    if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess ||
                        !isfinite(ms) || !(ms > 0.0f) || ms > 10000.0f) { ok = false; break; }
                    (mma ? b : a)[mode][pair] = (double)ms * 1000.0 / repetitions;
                }
            }
        }
        if (!ok) break;
        success = true;
    } while (false);
    /* Includes all partial-allocation, native mismatch and timing failures. */
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (bout) (void)cudaFree(bout);
    if (aout) (void)cudaFree(aout);
    if (xs) (void)cudaFree(xs);
    if (xq) (void)cudaFree(xq);
    if (weights) (void)cudaFree(weights);
    free(host_weights);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report, sizeof report, "hcUp[failed=%s]", failure); return report;
    }
    double ratio[4] = {}, old_us[4] = {}, new_us[4] = {}, max_upper = 0.0;
    unsigned min_wins = 7;
    bool prefer = true;
    for (unsigned mode = 0; mode < 4u; mode++) {
        double mean = 0.0, variance = 0.0;
        unsigned wins = 0;
        for (unsigned i = 0; i < 7u; i++) {
            old_us[mode] += a[mode][i]; new_us[mode] += b[mode][i];
            mean += b[mode][i] / a[mode][i]; wins += b[mode][i] < a[mode][i];
        }
        mean /= 7.0;
        for (unsigned i = 0; i < 7u; i++) {
            const double delta = b[mode][i] / a[mode][i] - mean;
            variance += delta * delta;
        }
        const double upper = mean + 3.0 * sqrt(variance / 6.0) / sqrt(7.0);
        ratio[mode] = new_us[mode] / old_us[mode];
        prefer = prefer && ratio[mode] < 0.98 && wins >= 6u && upper < 0.98;
        if (upper > max_upper) max_upper = upper;
        if (wins < min_wins) min_wins = wins;
        old_us[mode] /= 7.0; new_us[mode] /= 7.0;
    }
    hc_up_tuned_device = device;
    hc_up_prefer_mma = prefer;
    snprintf(report, sizeof report,
             "hcUp[l2=%d a=%.4g,%.4g,%.4g,%.4g b=%.4g,%.4g,%.4g,%.4g upper=%.3g win=%u use=%u]",
             l2, old_us[0], old_us[1], old_us[2], old_us[3],
             new_us[0], new_us[1], new_us[2], new_us[3], max_upper, min_wins, (unsigned)prefer);
    return report;
}
