/* Startup-only comparison of identical 24-group screening keys. Width is
 * registered by scratch initialization. All weights and inputs are private
 * synthetic data; none are retained or used for inference. */
static const char *mtp_screen_mma_tune(void) {
    static char report[192] = "scrMma[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_MTP_NO_SCREEN_MMA") || getenv("DS4_MTP_NO_SCREEN_MMA_TUNE") ||
        getenv("DS4_MTP_NO_FUSED_SCREEN_KEYS") || getenv("DS4_QWEN4EXP_NO_ROW_TILE") ||
        getenv("DS4_QWEN4EXP_PAIR_LANES_R2")) {
        snprintf(report, sizeof report, "scrMma[disabled]"); return report;
    }
    const uint32_t width = mtp_screen_registered_width;
    int device = -1, major = 0, l2 = 0;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (MTP_NATIVE_SCREEN_GROUPS != 24u || MTP_NATIVE_DIM != 2560u ||
        width < 4096u || width > 131072u || g_n_gpus != 1 ||
        cudaGetDevice(&device) != cudaSuccess || device != g_gpu[0].device_id ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || major < 8 ||
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device) != cudaSuccess ||
        l2 <= 0 || (uint64_t)width * 24u * 34u <= 2ull * (unsigned)l2 ||
        cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError(); return report;
    }
    const uint32_t tail = 276u, prefix = width - tail, vocab = width + 19u;
    const size_t weight_bytes = (size_t)vocab * 2720u, key_bytes = (size_t)width * 8u;
    unsigned char *weights = nullptr, *host_weights = nullptr;
    int8_t *xq = nullptr;
    float *xs = nullptr;
    uint64_t *old_keys = nullptr, *new_keys = nullptr, *old_host = nullptr, *new_host = nullptr;
    uint32_t *flag = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    bool success = false;
    const char *failure = "setup";
    double a[4][7] = {}, b[4][7] = {};
    do {
        host_weights = (unsigned char *)malloc(weight_bytes);
        old_host = (uint64_t *)malloc(key_bytes);
        new_host = (uint64_t *)malloc(key_bytes);
        if (!host_weights || !old_host || !new_host || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMalloc((void **)&weights, weight_bytes) != cudaSuccess ||
            cudaMalloc((void **)&xq, 2560u) != cudaSuccess ||
            cudaMalloc((void **)&xs, 320u) != cudaSuccess ||
            cudaMalloc((void **)&old_keys, key_bytes) != cudaSuccess ||
            cudaMalloc((void **)&new_keys, key_bytes) != cudaSuccess ||
            cudaMalloc((void **)&flag, 4u) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) break;
        uint32_t random = 0x393271abu;
        auto next = [&]() -> uint32_t {random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random;};
        for (size_t block = 0; block < weight_bytes / 34u; block++) {
            unsigned char *p = host_weights + block * 34u;
            const uint16_t scale = (uint16_t)(0x1000u | (next() & 0x83ffu));
            memcpy(p, &scale, 2);
            for (unsigned k = 2; k < 34u; k++) p[k] = (unsigned char)next();
        }
        if (cudaMemcpy(weights, host_weights, weight_bytes, cudaMemcpyHostToDevice) != cudaSuccess) break;
        auto input = [&](unsigned pattern) -> bool {
            int8_t x[2560]; float s[80];
            for (unsigned i = 0; i < 2560u; i++) x[i] = pattern == 0 ? 0 :
                pattern == 1 ? -128 : pattern == 2 ? 127 : (int8_t)next();
            for (unsigned i = 0; i < 80u; i++) {
                const float magnitude = pattern == 4 ? 1.0e-38f : pattern == 5 ? 1.0e10f : 0.003f;
                s[i] = (i & 1u) ? -magnitude : magnitude;
            }
            return cudaMemcpy(xq, x, sizeof x, cudaMemcpyHostToDevice) == cudaSuccess &&
                   cudaMemcpy(xs, s, sizeof s, cudaMemcpyHostToDevice) == cudaSuccess;
        };
        auto launch = [&](bool mma) -> bool {
            if (cudaMemsetAsync(flag, 0, 4u, stream) != cudaSuccess) return false;
            if (mma) {
                mtp_native_screen_mma_kernel<<<(width + 63u) / 64u, 128, 0, stream>>>(
                        new_keys, flag, weights, xq, xs, width, vocab, prefix, tail);
            } else {
                mtp_native_projection_kernel<true,true><<<(width + 3u) / 4u, 256, 0, stream>>>(
                        nullptr, weights, xq, xs, width, nullptr, vocab, prefix, tail, old_keys, flag);
            }
            return cudaGetLastError() == cudaSuccess;
        };
        bool ok = true;
        failure = "verify";
        for (unsigned pattern = 0; pattern < 6u && ok; pattern++) {
            uint32_t old_flag = 1, new_flag = 1;
            if (!input(pattern) || !launch(false) ||
                cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaMemcpy(old_host, old_keys, key_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(&old_flag, flag, 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                !launch(true) || cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaMemcpy(new_host, new_keys, key_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(&new_flag, flag, 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                old_flag || new_flag || memcmp(old_host, new_host, key_bytes)) ok = false;
        }
        if (!ok) break;
        failure = "timing";
        /* Every projection reads >2*L2 useful weight bytes at this registered
         * width. Four input patterns, seven AB/BA pairs each, sixteen complete
         * flag-reset/projection operations per leg. */
        for (unsigned pattern = 0; pattern < 4u && ok; pattern++) {
            if (!input(pattern) || !launch(false) || !launch(true) ||
                cudaStreamSynchronize(stream) != cudaSuccess) {ok = false; break;}
            for (unsigned pair = 0; pair < 7u && ok; pair++) {
                for (unsigned leg = 0; leg < 2u && ok; leg++) {
                    const bool mma = ((pair + leg) & 1u) != 0;
                    if (cudaEventRecord(begin, stream) != cudaSuccess) {ok = false; break;}
                    for (unsigned repeat = 0; repeat < 16u && ok; repeat++) ok = launch(mma);
                    if (!ok || cudaEventRecord(end, stream) != cudaSuccess ||
                        cudaEventSynchronize(end) != cudaSuccess) {ok = false; break;}
                    float ms = 0.0f;
                    if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess ||
                        !isfinite(ms) || !(ms > 0.0f) || ms > 10000.0f) {ok = false; break;}
                    (mma ? b : a)[pattern][pair] = (double)ms * 1000.0 / 16.0;
                }
            }
        }
        if (!ok) break;
        success = true;
    } while (false);
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (flag) (void)cudaFree(flag);
    if (new_keys) (void)cudaFree(new_keys);
    if (old_keys) (void)cudaFree(old_keys);
    if (xs) (void)cudaFree(xs);
    if (xq) (void)cudaFree(xq);
    if (weights) (void)cudaFree(weights);
    free(new_host); free(old_host); free(host_weights);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report, sizeof report, "scrMma[failed=%s]", failure); return report;
    }
    double ratios[4] = {}, sum_a = 0.0, sum_b = 0.0, max_upper = 0.0;
    unsigned min_wins = 7;
    bool prefer = true;
    for (unsigned pattern = 0; pattern < 4u; pattern++) {
        double sa = 0.0, sb = 0.0, mean = 0.0, variance = 0.0;
        unsigned wins = 0;
        for (unsigned i = 0; i < 7u; i++) {
            sa += a[pattern][i]; sb += b[pattern][i];
            mean += b[pattern][i] / a[pattern][i]; wins += b[pattern][i] < a[pattern][i];
        }
        mean /= 7.0;
        for (unsigned i = 0; i < 7u; i++) {const double delta = b[pattern][i] / a[pattern][i] - mean; variance += delta * delta;}
        const double upper = mean + 3.0 * sqrt(variance / 6.0) / sqrt(7.0);
        ratios[pattern] = sb / sa;
        prefer = prefer && ratios[pattern] < 0.98 && wins >= 6u && upper < 0.98;
        if (upper > max_upper) max_upper = upper;
        if (wins < min_wins) min_wins = wins;
        sum_a += sa; sum_b += sb;
    }
    mtp_screen_tuned_device = device; mtp_screen_tuned_width = width; mtp_screen_prefer_mma = prefer;
    snprintf(report, sizeof report,
             "scrMma[w=%u a_us=%.4g b_us=%.4g r=%.3g,%.3g,%.3g,%.3g upper=%.3g win=%u use=%u]",
             width, sum_a / 28.0, sum_b / 28.0, ratios[0], ratios[1], ratios[2], ratios[3], max_upper, min_wins, (unsigned)prefer);
    return report;
}
