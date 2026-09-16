/* Startup-only, shape-specific integer-sort autotuning. Called by hw_limits
 * before the resident socket is bound. Synthetic IDs, no model-weight access.
 * No score, request, token history or benchmark state enters this decision. */
static int mtp_id_tuned_device = -1;
static bool mtp_id_prefer_block18 = false;

static const char *mtp_native_id_sort_tune(void) {
    static char report[128] = "idSort18[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_MTP_NO_ID_AUTOTUNE") || getenv("DS4_MTP_NO_BLOCK_ID_SORT")) {
        snprintf(report,sizeof report,"idSort18[disabled]");return report;
    }
    int device = -1;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (g_n_gpus != 1 || cudaGetDevice(&device) != cudaSuccess ||
        device != g_gpu[0].device_id ||
        cudaStreamIsCapturing(stream,&capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return report;

    uint64_t *keys = nullptr;
    uint32_t *ids = nullptr, *id_tmp = nullptr;
    void *temporary = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    size_t temporary_bytes = 0;
    const char *failure = "setup";
    bool success = false;
    double a[7] = {}, b[7] = {};
    do {
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cub::DeviceRadixSort::SortKeys(nullptr,temporary_bytes,
                (const uint32_t *)nullptr,(uint32_t *)nullptr,
                MTP_NATIVE_CAP,0,18,stream) != cudaSuccess) break;
        if (!temporary_bytes) temporary_bytes = 1;
        if (cudaMalloc((void **)&keys,MTP_NATIVE_CAP*8ull) != cudaSuccess ||
            cudaMalloc((void **)&ids,MTP_NATIVE_CAP*4ull) != cudaSuccess ||
            cudaMalloc((void **)&id_tmp,MTP_NATIVE_CAP*4ull) != cudaSuccess ||
            cudaMalloc(&temporary,temporary_bytes) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) break;
        auto launch = [&](bool block) -> bool {
            if (block) {
                mtp_native_unpack_sort_ids<<<1,256,0,stream>>>(ids,keys,18);
                return cudaGetLastError() == cudaSuccess;
            }
            mtp_native_unpack_ids<<<8,256,0,stream>>>(id_tmp,keys);
            if (cudaGetLastError() != cudaSuccess) return false;
            size_t bytes = temporary_bytes;
            return cub::DeviceRadixSort::SortKeys(temporary,bytes,id_tmp,ids,
                MTP_NATIVE_CAP,0,18,stream) == cudaSuccess;
        };
        uint64_t host_keys[MTP_NATIVE_CAP];
        uint32_t expected[MTP_NATIVE_CAP], got[MTP_NATIVE_CAP];
        bool verified = true;
        failure = "verify";
        for (unsigned pattern=0;pattern<3 && verified;pattern++) {
            for (unsigned i=0;i<MTP_NATIVE_CAP;i++) {
                const uint32_t id = pattern==0 ? i : pattern==1 ? 262143u-i
                    : (i*131071u+7919u)&262143u;
                expected[i]=id;
                host_keys[i]=((uint64_t)(i*1664525u+1013904223u)<<32)|(UINT32_MAX-id);
            }
            std::sort(expected,expected+MTP_NATIVE_CAP);
            if (cudaMemcpy(keys,host_keys,sizeof host_keys,cudaMemcpyHostToDevice) != cudaSuccess) {verified=false;break;}
            for (unsigned variant=0;variant<2;variant++) {
                if (!launch(variant!=0) ||
                    cudaMemcpy(got,ids,sizeof got,cudaMemcpyDeviceToHost) != cudaSuccess ||
                    memcmp(got,expected,sizeof got)) {verified=false;break;}
            }
        }
        if (!verified) break;
        failure = "warmup";
        for (unsigned i=0;i<4 && verified;i++) verified=launch(false)&&launch(true);
        if (!verified || cudaStreamSynchronize(stream) != cudaSuccess) break;
        failure = "timing";
        for (unsigned pair=0;pair<7 && verified;pair++) {
            for (unsigned leg=0;leg<2;leg++) {
                const bool block = ((pair+leg)&1u)!=0;
                if (cudaEventRecord(begin,stream) != cudaSuccess) {verified=false;break;}
                for (unsigned repeat=0;repeat<64 && verified;repeat++) verified=launch(block);
                if (!verified || cudaEventRecord(end,stream) != cudaSuccess ||
                    cudaEventSynchronize(end) != cudaSuccess) {verified=false;break;}
                float ms=0;
                if (cudaEventElapsedTime(&ms,begin,end) != cudaSuccess ||
                    !isfinite(ms) || !(ms>0.0f) || ms>64000.0f) {verified=false;break;}
                (block?b:a)[pair]=(double)ms*1000.0/64.0;
            }
        }
        if (!verified) break;
        success = true;
    } while (false);
    /* Every partial allocation/event path has the same cleanup. A failed
     * verification or measurement leaves inference on the ordinary sort. */
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (temporary) (void)cudaFree(temporary);
    if (id_tmp) (void)cudaFree(id_tmp);
    if (ids) (void)cudaFree(ids);
    if (keys) (void)cudaFree(keys);
    if (!success) {
        (void)cudaGetLastError();
        snprintf(report,sizeof report,"idSort18[failed=%s]",failure);
        return report;
    }
    double sum_a=0,sum_b=0,ratio_mean=0,ratio_variance=0;
    unsigned wins=0;
    for (unsigned i=0;i<7;i++) {sum_a+=a[i];sum_b+=b[i];ratio_mean+=b[i]/a[i];wins+=b[i]<a[i];}
    ratio_mean/=7.0;
    for (unsigned i=0;i<7;i++) {const double delta=b[i]/a[i]-ratio_mean;ratio_variance+=delta*delta;}
    const double ratio_sd=sqrt(ratio_variance/6.0);
    mtp_id_tuned_device=device;
    mtp_id_prefer_block18=sum_b<0.98*sum_a && wins>=6 && ratio_sd<0.05;
    snprintf(report,sizeof report,"idSort18[a_us=%.4g b_us=%.4g r_sd=%.3g win=%u use=%u]",
             sum_a/7.0,sum_b/7.0,ratio_sd,wins,(unsigned)mtp_id_prefer_block18);
    return report;
}
