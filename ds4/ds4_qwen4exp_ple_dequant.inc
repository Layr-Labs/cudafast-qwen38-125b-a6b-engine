/* ds4_qwen4exp_ple_dequant.inc -- parallel dequantization fan-out for the
 * PLE n-gram gather.
 *
 * Included by ds4_qwen4exp_graph.inc just before qwen4exp_graph_ple_gather,
 * after the gather pool (g_qw_ple_pool, qw_ple_gather_seg,
 * qw_ple_gather_worker, qw_ple_pool_init, QW_PLE_MAX_THREADS) and the PLE
 * host helpers (ds4_qwen4exp_ple_row, ds4_ple_dequant_iq4_nl) are defined.
 *
 * The gather's row-id derivation is sequential -- each token's ids are a
 * function of the n-gram history the previous token left -- but the rows
 * those ids name are independent reads of the mapped table and independent
 * dequantizations into disjoint slices of ple_rows_host.  This file fans
 * that second phase across the existing pool at decode widths, where the
 * gather otherwise runs serially on one core.
 *
 * A segment is marked dequant-only by a NULL tokens pointer: the dispatch
 * here sets tokens = NULL and reinterprets t0/t1 as a range of the flat
 * (token, head) index space [0, n_tokens * DS4_N_PLE_HEAD).  The gather
 * worker tests the pointer before its history warm-up -- which would
 * dereference tokens -- and runs only the dequant loop.  The token-range
 * dispatch always installs a non-NULL pointer, so the marker cannot
 * collide with a real gather segment. */

/* The gather's thread count, resolved once per process: the env override,
 * else the online CPUs capped at the pool width.  This runs on every
 * gather, and both getenv and sysconf(_SC_NPROCESSORS_ONLN) are libc calls
 * that walk external state -- the environ array and the online-CPU map --
 * to return a value that cannot change while the engine is open.  Caching
 * it removes two of those walks from the round. */
static long qw_ple_gather_threads(void) {
    static long cached = -1;
    if (cached < 0) {
        const char * const thr_env = getenv("DS4_QWEN4EXP_PLE_GATHER_THREADS");
        long v = 0;
        if (thr_env && thr_env[0]) {
            char *end = NULL;
            v = strtol(thr_env, &end, 10);
            if (end == thr_env || v < 0) v = 0;
        }
        if (v <= 0) {
            v = sysconf(_SC_NPROCESSORS_ONLN);
            if (v > 16) v = 16;
            if (v < 1) v = 1;
        }
        cached = v;
    }
    return cached;
}

/* Fan the dequantization of an already-computed id block over the pool.
 * The ids are sequential -- the n-gram history is -- but the rows they name
 * are independent reads and independent dequantizations, so the flat
 * (token, head) space splits across workers with each row landing in a
 * disjoint slice of ple_rows_host.  Returns 1 when the rows are all
 * written, -1 when a row id was out of range, and 0 when the fan-out did
 * not apply (pool off, too little work, or no workers) and the caller runs
 * its serial loop instead. */
static int qw_ple_dequant_parallel(ds4_qwen4exp_session *s,
                                   const ds4_qwen4exp_ple_table *ple,
                                   uint32_t n_tokens,
                                   uint32_t blocks_per_row) {
    const uint32_t n_rows = n_tokens * DS4_N_PLE_HEAD;
    long nthreads = qw_ple_gather_threads();
    if (nthreads <= 1 || n_rows < 8u) return 0;
    if ((uint32_t)nthreads > n_rows) nthreads = (long)n_rows;
    if (nthreads > QW_PLE_MAX_THREADS) nthreads = QW_PLE_MAX_THREADS;

    qw_ple_pool_init();
    if (!g_qw_ple_pool.initialized) return 0;
    if ((long)g_qw_ple_pool.n_threads < nthreads)
        nthreads = (long)g_qw_ple_pool.n_threads;
    if (nthreads <= 1) return 0;

    const uint32_t per =
        (n_rows + (uint32_t)nthreads - 1u) / (uint32_t)nthreads;
    uint32_t actual_threads = 0;
    for (long i = 0; i < nthreads; i++) {
        const uint32_t t0 = (uint32_t)i * per;
        if (t0 >= n_rows) break;
        actual_threads++;
        uint32_t t1 = t0 + per;
        if (t1 > n_rows) t1 = n_rows;
        g_qw_ple_pool.segs[i].ple = ple;
        g_qw_ple_pool.segs[i].s = s;
        g_qw_ple_pool.segs[i].tokens = NULL;
        g_qw_ple_pool.segs[i].n_tokens = n_tokens;
        g_qw_ple_pool.segs[i].t0 = t0;
        g_qw_ple_pool.segs[i].t1 = t1;
        g_qw_ple_pool.segs[i].blocks_per_row = blocks_per_row;
        g_qw_ple_pool.segs[i].ok = 0;
    }
    if (actual_threads <= 1) return 0;

    pthread_mutex_lock(&g_qw_ple_pool.mutex);
    g_qw_ple_pool.active_workers = actual_threads - 1u;
    g_qw_ple_pool.completed_workers = 0;
    g_qw_ple_pool.generation++;
    pthread_cond_broadcast(&g_qw_ple_pool.work_cond);
    pthread_mutex_unlock(&g_qw_ple_pool.mutex);

    /* Master thread runs segment 0 concurrently with workers. */
    qw_ple_gather_worker(&g_qw_ple_pool.segs[0]);

    pthread_mutex_lock(&g_qw_ple_pool.mutex);
    while (g_qw_ple_pool.completed_workers < g_qw_ple_pool.active_workers) {
        pthread_cond_wait(&g_qw_ple_pool.done_cond, &g_qw_ple_pool.mutex);
    }
    pthread_mutex_unlock(&g_qw_ple_pool.mutex);

    for (uint32_t i = 0; i < actual_threads; i++) {
        if (g_qw_ple_pool.segs[i].ok != 1) return -1;
    }
    return 1;
}
