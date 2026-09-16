/* Exact partial selection. The pivot changes work, never the selected set:
 * use the compacted list only when it contains every key >= pivot and its
 * size is in [2048,4096]. Otherwise the caller retains the original full
 * radix sort. Valid original keys are nonzero and include ~ID for tie order. */
static constexpr uint32_t MTP_PARTIAL_SAMPLES = 1024u;
static constexpr uint32_t MTP_PARTIAL_LIMIT = 4096u;
static constexpr uint32_t MTP_PARTIAL_TARGET = 3072u;
static uint32_t mtp_partial_registered_width = 0u;
static uint32_t mtp_partial_tuned_width = 0u;
static int mtp_partial_tuned_device = -1;
static bool mtp_partial_prefer = false;

static bool mtp_partial_eligible(bool disjoint, uint32_t width,
                                 int device, int id_bits) {
    return disjoint && width >= MTP_PARTIAL_LIMIT && width <= MTP_NATIVE_MAX_WIDTH &&
        getenv("DS4_MTP_NO_PARTIAL_SELECT") == nullptr &&
        (getenv("DS4_MTP_FORCE_PARTIAL_SELECT") != nullptr ||
         (mtp_partial_prefer && width == mtp_partial_tuned_width &&
          device == mtp_partial_tuned_device && id_bits == 18));
}

static bool mtp_partial_count_fits(uint32_t count) {
    return count >= MTP_NATIVE_CAP && count <= MTP_PARTIAL_LIMIT;
}

__global__ static void mtp_partial_pivot(uint64_t *pivot,
                                       const uint64_t *keys, uint32_t width) {
    using Sort = cub::BlockRadixSort<uint64_t,256,4,cub::NullType,6>;
    __shared__ typename Sort::TempStorage storage;
    uint64_t items[4];
#pragma unroll
    for (unsigned i=0;i<4;i++) {
        const uint32_t sample=threadIdx.x*4u+i;
        items[i]=keys[(uint64_t)sample*width/MTP_PARTIAL_SAMPLES];
    }
    /* Native input rows have increasing original IDs. Blocked sample loads
     * retain that order, so the stable high-word sort also resolves ID ties.
     * Exact selection safety needs only a pivot, not this sampling property. */
    Sort(storage).SortDescendingBlockedToStriped(items,32,64);
    const uint32_t rank=(MTP_PARTIAL_TARGET*MTP_PARTIAL_SAMPLES+width-1u)/width-1u;
#pragma unroll
    for (unsigned i=0;i<4;i++)
        if (threadIdx.x+i*256u==rank) *pivot=items[i];
}

__global__ static void mtp_partial_filter(uint64_t *selected, uint32_t *count,
        const uint64_t *keys, const uint64_t *pivot, uint32_t width) {
    using Scan = cub::BlockScan<uint32_t,256>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ uint32_t base;
    const uint32_t at=blockIdx.x*256u+threadIdx.x;
    const uint64_t key=at<width?keys[at]:0;
    const uint32_t take=at<width && key>=*pivot;
    uint32_t offset,total;
    Scan(storage).ExclusiveSum(take,offset,total);
    if (threadIdx.x==0u) base=atomicAdd(count,total);
    __syncthreads();
    if (take && base+offset<MTP_PARTIAL_LIMIT) selected[base+offset]=key;
}

__global__ static void mtp_partial_sort_ids(uint32_t *ids,
        const uint64_t *selected, uint32_t count, int id_bits) {
    static_assert(MTP_NATIVE_CAP==256u*8u,"fixed output tile");
    using Keys = cub::BlockRadixSort<uint64_t,256,16,cub::NullType,6>;
    using Ids = cub::BlockRadixSort<uint32_t,256,8,cub::NullType,6>;
    __shared__ union Storage {
        typename Keys::TempStorage keys;
        typename Ids::TempStorage ids;
    } storage;
    uint64_t items[16];
#pragma unroll
    for (unsigned i=0;i<16;i++) {
        const uint32_t at=threadIdx.x+i*256u;
        items[i]=at<count?selected[at]:0;
    }
    Keys(storage.keys).SortDescendingBlockedToStriped(items);
    uint32_t original[8];
#pragma unroll
    for (unsigned i=0;i<8;i++) original[i]=UINT32_MAX-(uint32_t)items[i];
    __syncthreads();
    Ids(storage.ids).SortBlockedToStriped(original,0,id_bits);
#pragma unroll
    for (unsigned i=0;i<8;i++) ids[threadIdx.x+i*256u]=original[i];
}
