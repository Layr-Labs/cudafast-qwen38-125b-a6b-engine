/* Bounded startup comparison of exact selection strategies. The registered
 * width comes from head scratch allocation, before the resident socket binds.
 * All keys are synthetic; no model weight, request or scoring state is read. */
static const char *mtp_partial_tune(void) {
    static char report[192] = "partSel[unmeasured]";
    static bool tried = false;
    if (tried) return report;
    tried = true;
    if (getenv("DS4_MTP_NO_PARTIAL_SELECT") || getenv("DS4_MTP_NO_PARTIAL_TUNE")) {
        snprintf(report,sizeof report,"partSel[disabled]"); return report;
    }
    const uint32_t width = mtp_partial_registered_width;
    int device = -1;
    cudaStreamCaptureStatus capture;
    const cudaStream_t stream = cuda_decode_stream();
    if (width < MTP_PARTIAL_LIMIT || width > 131072u || g_n_gpus != 1 ||
        cudaGetDevice(&device) != cudaSuccess || device != g_gpu[0].device_id ||
        cudaStreamIsCapturing(stream,&capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return report;
    uint64_t *keys = nullptr, *sorted = nullptr, *meta = nullptr;
    uint64_t *host = nullptr, *reference = nullptr;
    uint32_t *ids = nullptr, *id_tmp = nullptr;
    void *temporary = nullptr;
    size_t temporary_bytes = 0;
    cudaEvent_t begin = nullptr, end = nullptr;
    const char *failure = "setup";
    bool success = false, prefer = true;
    double a[4][7] = {}, b[4][7] = {};
    do {
        size_t score_bytes = 0, id_bytes = 0;
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cub::DeviceRadixSort::SortKeysDescending(nullptr,score_bytes,
                (const uint64_t *)nullptr,(uint64_t *)nullptr,width,32,64,stream) != cudaSuccess ||
            cub::DeviceRadixSort::SortKeys(nullptr,id_bytes,
                (const uint32_t *)nullptr,(uint32_t *)nullptr,MTP_NATIVE_CAP,0,18,stream) != cudaSuccess) break;
        temporary_bytes = std::max(score_bytes,id_bytes);
        if (!temporary_bytes) temporary_bytes = 1;
        if (cudaMalloc((void **)&keys,(size_t)width*8u) != cudaSuccess ||
            cudaMalloc((void **)&sorted,(size_t)width*8u) != cudaSuccess ||
            cudaMalloc((void **)&ids,MTP_NATIVE_CAP*4u) != cudaSuccess ||
            cudaMalloc((void **)&id_tmp,MTP_NATIVE_CAP*4u) != cudaSuccess ||
            cudaMalloc((void **)&meta,16u) != cudaSuccess ||
            cudaMalloc(&temporary,temporary_bytes) != cudaSuccess ||
            cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) break;
        host = (uint64_t *)malloc((size_t)width*8u);
        reference = (uint64_t *)malloc((size_t)width*8u);
        if (!host || !reference) break;
        auto make_keys = [&](unsigned pattern) -> bool {
            uint32_t random = 0x64291f5bu;
            for (uint32_t i=0;i<width;i++) {
                random ^= random<<13; random ^= random>>17; random ^= random<<5;
                uint32_t high = 0x80000000u+(random&0x7f7fffffu);
                if (pattern==1 || pattern==5) high=0x80000000u;
                if (pattern==2) high=0x80000000u+i;
                if (pattern==3) high=0x90000000u-i;
                if (pattern==4) high=0xc1000000u;
                if (pattern==6) high=(i&1u)?0x00800000u:0xff7fffffu;
                const uint32_t id=i<width-256u?i:262144u-(width-i);
                host[i]=((uint64_t)high<<32)|(UINT32_MAX-id);
            }
            if (pattern==4 || pattern==5)
                for (uint64_t i=0;i<MTP_PARTIAL_SAMPLES;i++) {
                    const uint32_t at=(uint32_t)(i*width/MTP_PARTIAL_SAMPLES);
                    host[at]=(host[at]&UINT32_MAX)|
                        ((uint64_t)(pattern==4?0x80000000u:0xc1000000u)<<32);
                }
            for (uint32_t i=0;i<width;i++)
                if (!i || i>=width-256u) host[i]|=0xffffffff00000000ull;
            return cudaMemcpy(keys,host,(size_t)width*8u,cudaMemcpyHostToDevice)==cudaSuccess;
        };
        auto launch = [&](bool partial) -> bool {
            uint32_t *const flag=(uint32_t *)meta;
            if (cudaMemsetAsync(flag,0,8u,stream)!=cudaSuccess) return false;
            if (partial) {
                mtp_partial_pivot<<<1,256,0,stream>>>(meta+1u,keys,width);
                if (cudaGetLastError()!=cudaSuccess) return false;
                mtp_partial_filter<<<(width+255u)/256u,256,0,stream>>>(sorted,flag+1u,keys,meta+1u,width);
                if (cudaGetLastError()!=cudaSuccess) return false;
            }
            uint32_t status[2] = {};
            /* Both legs include the same native-stage flag read. The new leg
             * returns its count alongside it, not in an additional read. */
            if (cudaMemcpy(status,flag,sizeof status,cudaMemcpyDeviceToHost)!=cudaSuccess) return false;
            if (partial && mtp_partial_count_fits(status[1])) {
                mtp_partial_sort_ids<<<1,256,0,stream>>>(ids,sorted,status[1],18);
                return cudaGetLastError()==cudaSuccess;
            }
            size_t bytes=temporary_bytes;
            if (cub::DeviceRadixSort::SortKeysDescending(temporary,bytes,keys,sorted,width,32,64,stream)!=cudaSuccess) return false;
            mtp_native_unpack_ids<<<8,256,0,stream>>>(id_tmp,sorted);
            if (cudaGetLastError()!=cudaSuccess) return false;
            bytes=temporary_bytes;
            return cub::DeviceRadixSort::SortKeys(temporary,bytes,id_tmp,ids,MTP_NATIVE_CAP,0,18,stream)==cudaSuccess;
        };
        uint32_t expected[MTP_NATIVE_CAP], got[MTP_NATIVE_CAP];
        bool ok = true;
        failure = "verify";
        for (unsigned pattern=0;pattern<7u && ok;pattern++) {
            ok=make_keys(pattern);
            if (!ok) break;
            memcpy(reference,host,(size_t)width*8u);
            std::sort(reference,reference+width,[](uint64_t x,uint64_t y){return x>y;});
            for (unsigned i=0;i<MTP_NATIVE_CAP;i++) expected[i]=UINT32_MAX-(uint32_t)reference[i];
            std::sort(expected,expected+MTP_NATIVE_CAP);
            for (unsigned variant=0;variant<2u && ok;variant++)
                ok=launch(variant!=0) && cudaMemcpy(got,ids,sizeof got,cudaMemcpyDeviceToHost)==cudaSuccess &&
                    !memcmp(got,expected,sizeof got);
        }
        if (!ok) break;
        failure = "timing";
        for (unsigned pattern=0;pattern<4u && ok;pattern++) {
            ok=make_keys(pattern) && launch(false) && launch(true) && cudaStreamSynchronize(stream)==cudaSuccess;
            for (unsigned pair=0;pair<7u && ok;pair++) for (unsigned leg=0;leg<2u && ok;leg++) {
                const bool partial=((pair+leg)&1u)!=0;
                ok=cudaEventRecord(begin,stream)==cudaSuccess;
                for (unsigned repeat=0;repeat<8u && ok;repeat++) ok=launch(partial);
                float ms=0;
                ok=ok && cudaEventRecord(end,stream)==cudaSuccess && cudaEventSynchronize(end)==cudaSuccess &&
                    cudaEventElapsedTime(&ms,begin,end)==cudaSuccess && isfinite(ms) && ms>0.0f && ms<8000.0f;
                if (ok) (partial?b:a)[pattern][pair]=(double)ms*1000.0/8.0;
            }
        }
        if (!ok) break;
        success=true;
    } while (false);
    if (reference) free(reference);
    if (host) free(host);
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (temporary) (void)cudaFree(temporary);
    if (meta) (void)cudaFree(meta);
    if (id_tmp) (void)cudaFree(id_tmp);
    if (ids) (void)cudaFree(ids);
    if (sorted) (void)cudaFree(sorted);
    if (keys) (void)cudaFree(keys);
    if (!success) {
        (void)cudaGetLastError();snprintf(report,sizeof report,"partSel[failed=%s]",failure);return report;
    }
    double ratio[4]={}, mean_a=0,mean_b=0,worst_sd=0,worst_upper=0;
    unsigned least_wins=7;
    for (unsigned p=0;p<4u;p++) {
        double sa=0,sb=0,mean=0,variance=0;unsigned wins=0;
        for (unsigned i=0;i<7u;i++) {sa+=a[p][i];sb+=b[p][i];mean+=b[p][i]/a[p][i];wins+=b[p][i]<a[p][i];}
        mean/=7.0;
        for (unsigned i=0;i<7u;i++) {const double d=b[p][i]/a[p][i]-mean;variance+=d*d;}
        const double sd=sqrt(variance/6.0);
        /* Require the measured saving to exceed its scatter. A fixed SD
         * cutoff rejects even consistently large wins; this margin scales
         * with the effect instead. This is a conservative selection rule,
         * not a confidence claim about independent model-time samples. */
        const double upper=mean+3.0*sd/sqrt(7.0);
        if (sd>worst_sd) worst_sd=sd;
        if (upper>worst_upper) worst_upper=upper;
        if (wins<least_wins) least_wins=wins;
        ratio[p]=sb/sa;mean_a+=sa/28.0;mean_b+=sb/28.0;
        prefer=prefer && sb<0.98*sa && wins>=6 && upper<0.98;
    }
    mtp_partial_tuned_width=width;mtp_partial_tuned_device=device;mtp_partial_prefer=prefer;
    snprintf(report,sizeof report,"partSel[w=%u a_us=%.4g b_us=%.4g r=%.3g,%.3g,%.3g,%.3g sd=%.3g upper=%.3g win=%u use=%u]",
        width,mean_a,mean_b,ratio[0],ratio[1],ratio[2],ratio[3],worst_sd,worst_upper,least_wins,(unsigned)prefer);
    return report;
}
