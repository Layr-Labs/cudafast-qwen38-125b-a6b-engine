/*
 * End-to-end test of the qwen4exp serial graph on the reduced-layer synthetic
 * GGUF set that tools/qwen4exp_synthetic_gguf.py writes: 4 blocks at production
 * per-layer shapes, three GDN and one QSA, split over EIGHT shards with shard 0
 * metadata-only, the way the production artifact is laid out.  Eight and not
 * three because eight puts a file boundary INSIDE a block's tensor groups --
 * the hyper-connection group of blocks 1 and 2, the gated-delta-net group of
 * block 2 alone, and the expert group of blocks 0 and 3 -- and three left whole
 * groups together, where a weight resolved through a sibling's mapping still
 * read the right bytes.
 *
 * What it asserts:
 *   1. the forward runs on the synthetic file, through the real embedding
 *      gather, both hyper-connection mixers, the GDN blocks, the QSA block and
 *      its indexer, the routed MoE, the shared expert, the final mixer and the
 *      LM head, and the real PLE block: its kernels are written on both
 *      backends and the fixture carries the n-gram hash constants they need.
 *   1b. the numbers are FINITE and not all zero.  The synthetic payload decodes
 *      to finite, small weights, so every check below compares real numbers;
 *      a checkpoint whose f16 block scales decoded to infinities made the whole
 *      forward NaN, and NaN compares equal to itself bit for bit.
 *   2. a serial free run is bit-exact over three runs, in the sampled logits
 *      AND in the pre-final-mixer hyper-connection rows, which is the stream
 *      the native MTP head reads and the one L9's verifier needs.
 *   2b. those rows are the content the final mixer read: the graph copies the
 *      stream inside the command batch immediately before the last mixer call
 *      and the test compares those bytes with what the public accessor returns
 *      afterwards, which catches a mixer writing through the buffer as well as
 *      an accessor handing back a copy.
 *   2d. the same model written as ONE shard gives byte-identical logits, rows
 *      and slice order.  Every weight must resolve against the mapping that
 *      actually holds it; reading the right offset out of a sibling tensor's
 *      mapping yields plausible numbers and refuses nowhere, so a
 *      split-versus-single comparison is the only thing that catches it.
 *   2c. the slices ran in the model's order.  The graph records each slice at
 *      its own call site and the test asserts the whole sequence, so a slice
 *      that is dropped, duplicated or moved fails -- notably the PLE block,
 *      which MLX applies at the HEAD of its layer and which passed the old
 *      "the forward returned true" check from either position.
 *   3. the memory guard refuses on a fake low reading BEFORE it allocates any
 *      device buffer -- proved by the session allocator's counter, not by the
 *      order of lines in the source.
 *
 * The fourth assertion of the slice, that `ds4 -m <shard 00001>` opens the same
 * file through the production entry with no test hook, is in the shell driver
 * next to this file: it needs the shipped binary, not this one.
 *
 * DEVICE MEMORY.  This test is the heaviest qwen4exp target: the reduced
 * synthetic model is ~6.9 GiB of device memory on CUDA, and the session's own
 * plan prints 6.48 GiB resident weights plus ~0.01 GiB of session state (the
 * loader's plan says 6.68 GiB including the SSD-resident n-gram table).  The
 * 512-expert geometry is fixture-fixed, so this does not shrink.  A per-target
 * memory allowance for the test runner wants that figure, not the size of the
 * files on disk, which are mostly holes.
 *
 * The two phases are two PROCESSES for the same reason: Metal keeps a closed
 * model's views resident for the life of the process, so opening all three
 * fixtures in one run exhausts GPU memory on a 24 GiB machine.
 *
 * The synthetic context is small, so the QSA block runs its DENSE causal path:
 * the indexer only engages past 2048 visible keys.  The sparse path is covered
 * by tests/test_qwen4exp_qsa.c at the production budget.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4.h"
#include "ds4_gpu.h"
/* For the depth envelope the speculative section sweeps. */
#include "ds4_qwen4exp_mtp.h"

/* Session plan mirror.  ds4_qwen4exp_graph.h is compiled into ds4.o, not into
 * this file; keep the field order in step with it. */
typedef struct {
    uint64_t gdn_recurrent_bytes;
    uint64_t gdn_conv_bytes;
    uint64_t qsa_kv_bytes;
    uint64_t qsa_indexer_bytes;
    uint64_t ple_state_bytes;
    uint32_t spec_snapshot_slots;
    uint64_t spec_snapshot_bytes;
    uint64_t activation_bytes;
    uint64_t total_bytes;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_gdn_layer;
    uint32_t n_qsa_layer;
} session_plan;

uint32_t ds4_qwen4exp_test_graph_run(const char *path, uint32_t n_ctx,
                                     const int32_t *tokens, uint32_t n_tokens,
                                     uint32_t runs, uint64_t free_bytes,
                                     float *logits_out, float *hyper_out,
                                     uint32_t *hyper_row_floats_out,
                                     int *hyper_is_mixer_input_out,
                                     uint8_t *trace_out, uint32_t *trace_len_out,
                                     uint32_t trace_cap, uint32_t *n_vocab_out);

/* Mirrors qwen4exp_slice in ds4_qwen4exp_graph.inc. */
enum {
    QW_SLICE_EMBED = 1,
    QW_SLICE_PLE,
    QW_SLICE_ATTN_MIX,
    QW_SLICE_GDN,
    QW_SLICE_QSA,
    QW_SLICE_ATTN_INJECT,
    QW_SLICE_FFN_MIX,
    QW_SLICE_MOE,
    QW_SLICE_FFN_INJECT,
    QW_SLICE_FINAL_MIX,
    QW_SLICE_HEAD,
};
uint64_t ds4_qwen4exp_test_graph_alloc_count(void);
int ds4_qwen4exp_test_layer_sums(double *out, int cap, int *rows_out);
int ds4_qwen4exp_test_text_probe(const char *model_path,
                                 const char *const *prompts, int n_prompts,
                                 int steps, int top_k, int *prompt_len_out,
                                 int *top_ids_out, float *top_logits_out,
                                 int *gen_out);
int ds4_qwen4exp_test_depth_table(const char *model_path,
                                  const char *head_path,
                                  const int *prompt_ids,
                                  const int *prompt_lens,
                                  int n_prompts,
                                  const int *depths, int n_depths,
                                  int gen_tokens, int max_prompt_len,
                                  int *tokens_out,
                                  int *rounds_out,
                                  int *committed_out,
                                  int *accepted_out,
                                  int *disagree_out,
                                  uint64_t *commit_hist_out,
                                  uint64_t *load_ns_out,
                                  uint64_t *decode_ns_out,
                                  uint64_t *prefill_ns_out,
                                  uint64_t *verify_ns_out,
                                  uint64_t *draft_ns_out,
                                  uint64_t *rollback_ns_out);

int ds4_qwen4exp_test_prompt_ladder(const char *model_path,
                                    const int *ids, int n_ids,
                                    const int *lengths,
                                    const uint32_t *batches,
                                    int n_cases, int steps_per_case, int top_k,
                                    int *batch_out, int *pos_out,
                                    int *top_ids_out, float *top_logits_out,
                                    int *gen_out, int gen_tokens);
int ds4_qwen4exp_test_support_probe(const char *model_path,
                                    const char *mtp_path);
int ds4_qwen4exp_test_open_with_support(const char *self_exe,
                                        const char *model_path,
                                        const char *mtp_path,
                                        int *is_nextn_out);
int ds4_qwen4exp_test_collect_tensor_spans(const char *path, uint64_t *nspan_out,
                                           uint64_t *bytes_out,
                                           uint64_t *skipped_ssd_out,
                                           uint32_t *shards_seen_out,
                                           int *sorted_ok_out,
                                           int *within_shard_ok_out,
                                           int *q8_ranges_ok_out,
                                           uint32_t *shard0_tensors_out);
int ds4_qwen4exp_test_head_block_run(const char *target_path,
                                     const char *head_path, uint32_t n_tokens,
                                     int *ran_out, int *deterministic_out,
                                     int *changed_out);
int ds4_qwen4exp_test_head_lm_head_identity(const char *target_path,
                                            uint32_t n_rows,
                                            int *one_row_same_out,
                                            int *row_invariant_out);
int ds4_qwen4exp_test_session_serial(const char *path, const int *prompt,
                                     int prompt_len, int n_gen, int *first_out,
                                     int *tokens_out, int *top_ok_out,
                                     int *reset_first_out,
                                     int *batch_refused_out,
                                     int *pos_after_sync_out,
                                     int *pos_after_gen_out,
                                     int *tape_len_out,
                                     int *tape_tail_ok_out);
int ds4_qwen4exp_test_session_spec(const char *path, const char *head_path,
                                   const int *prompt, int prompt_len,
                                   int n_gen, int draft_tokens,
                                   const int *forced, int forced_len,
                                   int *tokens_out, int *n_out,
                                   int *pos_after_gen_out, int *tape_len_out,
                                   int *tape_tail_ok_out,
                                   int *rounds_out, int *accepted_out,
                                   int *commit_deep_out, int *counters_ok_out,
                                   int *disagreements_out,
                                   int *pending_before_out,
                                   int *pending_after_invalidate_out,
                                   int *pending_after_sync_out,
                                   int *pending_after_eval_out);
int ds4_qwen4exp_test_prefill_chunk_argmax(const char *path, const int *prompt,
                                           int prompt_len, uint32_t chunk,
                                           int *pos_out, float *logits_out,
                                           int logits_cap, double *state_out,
                                           int state_cap, int fixed_batch,
                                           int presync);
int ds4_qwen4exp_test_batch_invariance(const char *path, const int *prompt,
                                       int prompt_len, int second_token,
                                       uint32_t wide_rows,
                                       int *argmax_two_out, int *argmax_one_out,
                                       double *max_abs_out, double *hc_worst_out,
                                       double *content_worst_out,
                                       double *hc_scale_out,
                                       double *seq_worst_out,
                                       double *state_batched,
                                       double *state_sequential,
                                       int state_cap);
int ds4_qwen4exp_test_width_carry(const char *path, const int *prompt,
                                  int prompt_len, int seed_token,
                                  uint32_t width, uint32_t n_decode,
                                  int ctx_size,
                                  double *prefill_worst,
                                  double *decode_worst,
                                  char *first_object, size_t first_object_len,
                                  double *first_object_diff,
                                  int *n_objects_out,
                                  int *n_objects_differ_out);
uint32_t ds4_qwen4exp_test_model_vocab(const char *path);
int ds4_qwen4exp_test_model_reopen_registry(const char *path, int reps,
                                            int *max_after_close,
                                            int *same_address_reopens);
int ds4_qwen4exp_test_cli_first_token(const char *path,
                                      const char *prompt_text,
                                      int         prompt_cap,
                                      int        *prompt_len_out,
                                      int        *sent_len_out,
                                      int        *sampled_out,
                                      int        *argmax_out,
                                      int        *is_stop_out,
                                      int        *is_stop_nothink_out);
int ds4_qwen4exp_test_session_glm_unreachable(const char *path, int *is_glm_out,
                                              int *glm_graph_ready_out,
                                              int *is_qwen4exp_out,
                                              int *spec_routed_out,
                                              int *entry_refused_out,
                                              int *entry_total_out);
bool ds4_qwen4exp_test_graph_session_open(const char *path, uint32_t n_ctx,
                                          uint32_t n_batch,
                                          uint64_t bind_free_bytes,
                                          uint64_t session_free_bytes,
                                          uint64_t already_resident_bytes,
                                          session_plan *plan_out);
#if !defined(__APPLE__)
int ds4_qwen4exp_test_shard_cache_policy(const char *path,
                                         uint64_t *required_out,
                                         uint64_t *streamed_out,
                                         uint32_t *required_shards_out,
                                         uint32_t *stream_only_shards_out);
#endif

static int g_checks = 0;
static int g_failures = 0;

static bool all_finite(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!(v[i] > -3.0e38f && v[i] < 3.0e38f)) return false;
    }
    return true;
}

static bool any_nonzero(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (v[i] != 0.0f) return true;
    }
    return false;
}

static void check(bool ok, const char *what) {
    g_checks++;
    if (ok) {
        printf("  ok    %s\n", what);
        return;
    }
    g_failures++;
    printf("  FAIL  %s\n", what);
}

int main(int argc, char **argv) {
    /* Two phases, two processes.  Metal keeps every mapped model view resident
     * for the life of the process and never releases a closed model's, so
     * opening all three of these multi-GiB fixtures in one run exhausts GPU
     * memory on a 24 GiB machine.  The driver invokes this binary once per
     * phase instead. */
    /* Bring the backend up before anything allocates.  Metal initialises
     * lazily inside its first entry point, so this was invisible there; CUDA
     * does not, and an uninitialised device count makes every session buffer
     * allocation fail at the device_id >= g_n_gpus guard, which returns without
     * a word.  The kernel tests all do this (tests/test_qwen4exp_moe.c:819);
     * this one did not. */
    /* The support probe, re-entered through exec by the parent below.  It runs
     * BEFORE ds4_gpu_init(): ds4_engine_open() brings the backend up itself,
     * and this process exists precisely so that a fresh device is used. */
    if (argc == 4 && strcmp(argv[1], "--support-probe") == 0) {
        return ds4_qwen4exp_test_support_probe(argv[2], argv[3]);
    }

#if !defined(__APPLE__)
    if (argc == 3 && strcmp(argv[1], "--cache-policy") == 0) {
        uint64_t required = 0, streamed = 0;
        uint32_t required_shards = 0, stream_only_shards = 0;
        const int rc = ds4_qwen4exp_test_shard_cache_policy(
                argv[2], &required, &streamed,
                &required_shards, &stream_only_shards);
        check(rc == 0,
              "all cached required tensors reuse their existing device ranges");
        check(required > 0 && required_shards > 0,
              "the fixture exercises required tensors on device shards");
        check(streamed == 1,
              "only per_layer_token_embd.weight is exempt as SSD-resident");
        printf("cache policy: %llu required tensors on %u shards, %llu "
               "streamed tensor, %u stream-only shards\n",
               (unsigned long long)required, required_shards,
               (unsigned long long)streamed, stream_only_shards);
        return g_failures ? 1 : 0;
    }
#endif

    /* --textprobe <model> <prompt file>: one prompt per LINE, tokenized by the
     * engine's own tokenizer and run through the same harness loop the ladder
     * uses.  This is the control: prompts that succeeded through the CLI are
     * run here, so the harness loop is either exonerated or convicted. */
    if (argc == 4 && strcmp(argv[1], "--textprobe") == 0) {
        enum { MAXP = 16, STEPS = 12, TOPK = 8 };
        static char *prompts[MAXP];
        static char buf[MAXP][4096];
        int n_prompts = 0;
        FILE *pf = fopen(argv[3], "r");
        if (!pf) { fprintf(stderr, "textprobe: cannot open %s\n", argv[3]); return 1; }
        while (n_prompts < MAXP && fgets(buf[n_prompts], sizeof(buf[0]), pf)) {
            size_t L = strlen(buf[n_prompts]);
            while (L && (buf[n_prompts][L-1] == '\n' || buf[n_prompts][L-1] == '\r')) {
                buf[n_prompts][--L] = '\0';
            }
            if (L == 0) continue;
            prompts[n_prompts] = buf[n_prompts];
            n_prompts++;
        }
        fclose(pf);
        printf("textprobe: %d prompts\n", n_prompts);

        static int   plen[MAXP];
        static int   tids[MAXP * STEPS * TOPK];
        static float tlog[MAXP * STEPS * TOPK];
        static int   gen[MAXP * STEPS];
        memset(gen, 0, sizeof(gen));
        const int rc = ds4_qwen4exp_test_text_probe(
                argv[2], (const char *const *)prompts, n_prompts, STEPS, TOPK,
                plen, tids, tlog, gen);
        if (rc != 0) { fprintf(stderr, "textprobe failed: %d\n", rc); return 1; }
        for (int c = 0; c < n_prompts; c++) {
            printf("\n=== [%s] prompt_tokens=%d\n", prompts[c], plen[c]);
            for (int st = 0; st < STEPS; st++) {
                printf("  step %2d top-%d:", st, TOPK);
                for (int i = 0; i < TOPK; i++) {
                    const int at = (c * STEPS + st) * TOPK + i;
                    printf(" %d/%.3f", tids[at], (double)tlog[at]);
                }
                printf("\n");
            }
            printf("  greedy ids:");
            for (int st = 0; st < STEPS; st++) printf(" %d", gen[c * STEPS + st]);
            printf("\n");
        }
        /* Per-layer residual sums from the LAST forward, when the dump is on.
         * Set DS4_QWEN4EXP_LAYER_DUMP=1 and give ONE prompt, so the last
         * forward is that prompt's final decode step. */
        {
            static double sums[128];
            int rows = 0;
            const int n = ds4_qwen4exp_test_layer_sums(sums, 128, &rows);
            if (n > 0) {
                printf("\nper-layer sum|hyper| over %d row(s):\n", rows);
                for (int i = 0; i < n; i++) {
                    printf("  layer %2d  %.6e\n", i, sums[i]);
                }
            }
        }
        return 0;
    }

    /* --depth-table <model shard 1> <head.gguf> <token-id file> [depths]:
     * ONE model load, one row per (depth, prompt).  The prompt file holds one
     * prompt per LINE as whitespace-separated token ids, so three golden
     * prompts are three lines.  `depths` defaults to 0,1,2,3.
     *
     * The point of the row layout is that every row shares a load: on the
     * 125B a load is minutes, and a table whose rows came from separate loads
     * would carry the load's variance into the comparison the table exists to
     * make. */
    if (argc >= 5 && strcmp(argv[1], "--depth-table") == 0) {
        enum { MAX_PROMPTS = 8, MAX_LEN = 4096, MAX_DEPTHS = 8, GEN = 64 };
        static int  ids[MAX_PROMPTS * MAX_LEN];
        static int  lens[MAX_PROMPTS];
        int n_prompts = 0;
        FILE *fp = fopen(argv[4], "r");
        if (!fp) {
            fprintf(stderr, "depth-table: cannot open %s\n", argv[4]);
            return 1;
        }
        char *line = NULL;
        size_t cap = 0;
        while (n_prompts < MAX_PROMPTS && getline(&line, &cap, fp) > 0) {
            int n = 0;
            const char *p = line;
            while (*p && n < MAX_LEN) {
                char *end = NULL;
                const long v = strtol(p, &end, 10);
                if (end == p) break;
                ids[(size_t)n_prompts * MAX_LEN + n] = (int)v;
                n++;
                p = end;
            }
            if (n > 0) { lens[n_prompts] = n; n_prompts++; }
        }
        free(line);
        fclose(fp);
        if (n_prompts == 0) {
            fprintf(stderr, "depth-table: no prompts in %s\n", argv[4]);
            return 1;
        }

        static int depths[MAX_DEPTHS] = { 0, 1, 2, 3 };
        int n_depths = 4;
        if (argc >= 6) {
            n_depths = 0;
            const char *p = argv[5];
            while (*p && n_depths < MAX_DEPTHS) {
                depths[n_depths++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        printf("depth-table: %d prompts, %d depths, %d tokens each\n",
               n_prompts, n_depths, GEN);

        const int rows = n_depths * n_prompts;
        static int tokens[MAX_DEPTHS * MAX_PROMPTS * GEN];
        static int rounds[MAX_DEPTHS * MAX_PROMPTS];
        static int committed[MAX_DEPTHS * MAX_PROMPTS];
        static int accepted[MAX_DEPTHS * MAX_PROMPTS];
        static int disagree[MAX_DEPTHS * MAX_PROMPTS];
        static uint64_t hist[MAX_DEPTHS * MAX_PROMPTS *
                             (DS4_QWEN4EXP_MTP_MAX_COMMIT + 1)];
        static uint64_t dec_ns[MAX_DEPTHS * MAX_PROMPTS];
        static uint64_t pre_ns[MAX_DEPTHS * MAX_PROMPTS];
        static uint64_t ver_ns[MAX_DEPTHS * MAX_PROMPTS];
        static uint64_t dr_ns[MAX_DEPTHS * MAX_PROMPTS];
        static uint64_t rb_ns[MAX_DEPTHS * MAX_PROMPTS];
        uint64_t load_ns = 0;
        memset(tokens, 0, sizeof(tokens));

        const int rc = ds4_qwen4exp_test_depth_table(
                argv[2], argv[3], ids, lens, n_prompts, depths, n_depths,
                GEN, MAX_LEN, tokens, rounds, committed, accepted, disagree,
                hist, &load_ns, dec_ns, pre_ns, ver_ns, dr_ns, rb_ns);
        if (rc != 0) {
            fprintf(stderr, "depth-table failed: %d\n", rc);
            return 1;
        }

        printf("\nload %.1f s\n", (double)load_ns / 1e9);
        printf("\n%-5s %-6s %7s %7s %8s %9s %10s %8s %8s %8s %8s\n",
               "depth", "prompt", "rounds", "commit", "accepted",
               "tok/round", "decode t/s", "verify%", "draft%", "rollb%",
               "disagree");
        for (int r = 0; r < rows; r++) {
            const int d = depths[r / n_prompts];
            const int p = r % n_prompts;
            const double secs = (double)dec_ns[r] / 1e9;
            const double phase =
                (double)(ver_ns[r] + dr_ns[r] + rb_ns[r]);
            printf("%-5d %-6d %7d %7d %8d %9.2f %10.2f %7.1f%% %7.1f%% "
                   "%7.1f%% %8d\n",
                   d, p, rounds[r], committed[r], accepted[r],
                   rounds[r] ? (double)committed[r] / rounds[r] : 0.0,
                   secs > 0.0 ? (double)committed[r] / secs : 0.0,
                   phase > 0.0 ? 100.0 * (double)ver_ns[r] / phase : 0.0,
                   phase > 0.0 ? 100.0 * (double)dr_ns[r] / phase : 0.0,
                   phase > 0.0 ? 100.0 * (double)rb_ns[r] / phase : 0.0,
                   disagree[r]);
        }

        printf("\nacceptance histogram, rounds committing k tokens\n");
        for (int r = 0; r < rows; r++) {
            const int d = depths[r / n_prompts];
            printf("  depth %d prompt %d:", d, r % n_prompts);
            for (int k = 1; k <= DS4_QWEN4EXP_MTP_MAX_COMMIT; k++) {
                printf(" k=%d:%llu", k,
                       (unsigned long long)hist[(size_t)r *
                           (DS4_QWEN4EXP_MTP_MAX_COMMIT + 1) + k]);
            }
            printf("\n");
        }

        printf("\nper-round phase split, milliseconds\n");
        for (int r = 0; r < rows; r++) {
            const int d = depths[r / n_prompts];
            const double n = rounds[r] ? (double)rounds[r] : 1.0;
            printf("  depth %d prompt %d: verify %.3f  draft %.3f  "
                   "rollback %.3f  prefill %.1f\n",
                   d, r % n_prompts,
                   (double)ver_ns[r] / n / 1e6,
                   (double)dr_ns[r] / n / 1e6,
                   (double)rb_ns[r] / n / 1e6,
                   (double)pre_ns[r] / 1e6);
        }

        /* THE CONTRACT, checked here rather than only reported: every depth
         * must emit the depth-0 stream exactly.  Row 0 of each prompt is the
         * serial row only when depth 0 was asked for; when it was not there is
         * nothing to compare against and the streams are reported alone. */
        if (depths[0] == 0) {
            int bad = 0;
            for (int r = n_prompts; r < rows; r++) {
                const int p = r % n_prompts;
                for (int i = 0; i < GEN; i++) {
                    if (tokens[(size_t)r * GEN + i] !=
                        tokens[(size_t)p * GEN + i]) {
                        printf("  DIVERGES: depth %d prompt %d token %d: "
                               "serial %d, mtp %d\n",
                               depths[r / n_prompts], p, i,
                               tokens[(size_t)p * GEN + i],
                               tokens[(size_t)r * GEN + i]);
                        bad++;
                        break;
                    }
                }
            }
            printf("\n%s\n", bad ? "STREAMS DIVERGE"
                                 : "every depth emitted the serial stream");
        }

        printf("\nstreams\n");
        for (int r = 0; r < rows; r++) {
            printf("  depth %d prompt %d:", depths[r / n_prompts],
                   r % n_prompts);
            for (int i = 0; i < GEN; i++) {
                printf(" %d", tokens[(size_t)r * GEN + i]);
            }
            printf("\n");
        }
        return 0;
    }

    /* --ladder <model shard 1> <token-id file>: ONE model load, many sessions.
     * A separates prompt LENGTH from FORWARD COUNT; B holds the length and
     * varies the count.  Every rung reports three decode steps, because the
     * question is whether decode advances after the prompt, not whether the
     * prompt's own first token is right. */
    if (argc >= 4 && strcmp(argv[1], "--ladder") == 0) {
        enum { MAX_IDS = 4096, CASES = 9, STEPS = 3, TOPK = 8, GEN = 16 };
        static int ids[MAX_IDS];
        int n_ids = 0;
        FILE *fp = fopen(argv[3], "r");
        if (!fp) { fprintf(stderr, "ladder: cannot open %s\n", argv[3]); return 1; }
        while (n_ids < MAX_IDS && fscanf(fp, "%d", &ids[n_ids]) == 1) n_ids++;
        fclose(fp);
        printf("ladder: %d prompt token ids from %s\n", n_ids, argv[3]);

        /* A: one forward per rung up to 512, then two forwards. */
        static int      lengths[CASES] = { 64, 128, 256, 512, 513, 1024,
                                           64, 64, 64 };
        static uint32_t batches[CASES] = { 512u, 512u, 512u, 512u, 512u, 512u,
                                           /* B: fixed length 64 */
                                           64u, 8u, 2u };
        int n_cases = CASES;
        /* argv[4], if given, replaces the ladder with a comma-separated list of
         * lengths at one batch: the bracket run wants 8,12,16,24,32,48,64 and
         * nothing else. */
        if (argc >= 5) {
            n_cases = 0;
            const char *p = argv[4];
            while (*p && n_cases < CASES) {
                lengths[n_cases] = atoi(p);
                batches[n_cases] = 512u;
                n_cases++;
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            printf("ladder: %d explicit lengths at batch 512\n", n_cases);
        }
        static int   batch_used[CASES], pos_used[CASES];
        static int   top_ids[CASES * STEPS * TOPK];
        static float top_log[CASES * STEPS * TOPK];
        static int   gen[CASES * GEN];
        memset(gen, 0, sizeof(gen));

        const int rc = ds4_qwen4exp_test_prompt_ladder(
                argv[2], ids, n_ids, lengths, batches, n_cases, STEPS, TOPK,
                batch_used, pos_used, top_ids, top_log, gen, GEN);
        if (rc != 0) { fprintf(stderr, "ladder failed: %d\n", rc); return 1; }

        for (int c = 0; c < n_cases; c++) {
            const int fwd = batch_used[c] > 0
                          ? (lengths[c] + batch_used[c] - 1) / batch_used[c] : -1;
            printf("\n=== %s len=%d batch=%d forwards=%d pos_after_sync=%d\n",
                   (argc >= 5 || c < 6) ? "A" : "B", lengths[c],
                   batch_used[c], fwd,
                   pos_used[c]);
            for (int st = 0; st < STEPS; st++) {
                printf("  step %d top-%d:", st, TOPK);
                for (int i = 0; i < TOPK; i++) {
                    const int at = (c * STEPS + st) * TOPK + i;
                    printf(" %d/%.4f", top_ids[at], (double)top_log[at]);
                }
                printf("\n");
            }
            printf("  greedy ids:");
            for (int g = 0; g < GEN; g++) printf(" %d", gen[c * GEN + g]);
            printf("\n");
        }
        return 0;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "test_qwen4exp_graph: no GPU backend, skipping\n");
        return 0;
    }

    const bool mixed_phase = argc >= 2 && strcmp(argv[1], "--mixed") == 0;
    /* A third phase.  Each of these opens its own engine or session on top of
     * the fixtures the numeric phase already mapped, and Metal keeps a closed
     * model's views resident for the life of the process, so together they
     * exhaust GPU memory on 24 GiB.  Own process. */
    const bool session_phase = argc >= 2 && strcmp(argv[1], "--session") == 0;
    if ((mixed_phase && argc < 3) || (!mixed_phase && argc < 3)) {
        fprintf(stderr,
                "usage: %s <Q4_K 8-shard dir> <Q4_K 1-shard dir>\n"
                "       %s --mixed <mixed-quant dir>\n"
                "       %s --session <Q4_K 1-shard dir> <head dir>\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }

    char path[4096];
    char mixed_path[4096];
    char single_path[4096];
    if (mixed_phase) {
        snprintf(mixed_path, sizeof(mixed_path),
                 "%s/qw4x-00001-of-00003.gguf", argv[2]);
    } else if (session_phase) {
        /* Only the one-shard target and the head; the eight-shard set is the
         * numeric phase's and mapping it here would double the residency. */
        snprintf(single_path, sizeof(single_path), "%s/qw4x.gguf", argv[2]);
    } else {
        snprintf(path, sizeof(path), "%s/qw4x-00001-of-00008.gguf", argv[1]);
        snprintf(single_path, sizeof(single_path), "%s/qw4x.gguf", argv[2]);
    }

    /* A short prompt.  The ids stay well inside the synthetic vocabulary. */
    const int32_t tokens[] = {1, 7, 19, 42, 3, 3, 88, 5};
    const uint32_t n_tokens = (uint32_t)(sizeof(tokens) / sizeof(tokens[0]));
    const uint32_t n_ctx = 256;   /* a multiple of the indexer compress ratio */
    const uint32_t runs = 3;
    /* Sized from the writer's default vocabulary with room to spare; the run
     * hook reports the real one and the checks below hold it to it. */
    const uint32_t vocab_cap = 65536;
    /* Shared by both phases: the numeric phase overwrites n_vocab with what the
     * run hook reports; the session phase only needs a sane upper bound. */
    uint32_t n_vocab = vocab_cap;
    const size_t hyper_cap = (size_t)runs * n_tokens * 4u * 2560u;
    /* The synthetic set is 6.7 GiB resident, and the guard wants 10 GiB of
     * headroom, which a development Mac running a build does not have free.
     * Every check below therefore states the reading it wants; what is under
     * test is the guard's arithmetic and its ordering, not this machine. */
    const uint64_t plenty = 64ull * 1024ull * 1024ull * 1024ull;

    if (!mixed_phase && !session_phase) {
    printf("PLAN: the session plan is reported before anything is allocated\n");
    session_plan plan;
    memset(&plan, 0, sizeof(plan));
    const uint64_t before_refusal = ds4_qwen4exp_test_graph_alloc_count();

    /* One GiB free cannot hold the weights, the session and the headroom.  The
     * hook applies the fake reading after the bind, so this exercises the
     * session guard and not the loader's. */
    const bool refused = !ds4_qwen4exp_test_graph_session_open(
            path, n_ctx, n_tokens, plenty, 1024ull * 1024ull * 1024ull, 0u,
            &plan);
    const uint64_t after_refusal = ds4_qwen4exp_test_graph_alloc_count();

    /* ALREADY UPLOADED.  A device whose free-memory reading has already fallen
     * by the weight set -- CUDA's unified memory, once the engine has prepared
     * every tensor span -- must not be asked for those bytes twice.  A real
     * 76.86 GiB boot on a 118.6 GiB box read 40.17 GiB free and was refused a
     * session it had room for.
     *
     * Free memory here is BELOW the weights and above the session plus the
     * headroom, which is exactly the case that used to refuse. */
    const uint64_t weights_bytes = 7ull * 1024ull * 1024ull * 1024ull;
    const uint64_t after_upload = 11ull * 1024ull * 1024ull * 1024ull;
    session_plan plan_up;
    memset(&plan_up, 0, sizeof(plan_up));
    const bool opened_when_uploaded = ds4_qwen4exp_test_graph_session_open(
            path, n_ctx, n_tokens, plenty, after_upload, weights_bytes,
            &plan_up);
    /* NEGATIVE CONTROL: the same reading with nothing uploaded still refuses,
     * so the case above passes because of the accounting and not because the
     * reading was generous. */
    session_plan plan_no;
    memset(&plan_no, 0, sizeof(plan_no));
    const bool refused_when_not_uploaded = !ds4_qwen4exp_test_graph_session_open(
            path, n_ctx, n_tokens, plenty, after_upload, 0u, &plan_no);

    check(opened_when_uploaded,
          "a session opens when the weights are already on the device and the "
          "free reading no longer covers them");
    check(refused_when_not_uploaded,
          "the same free reading still refuses when nothing was uploaded");

    check(plan.n_gdn_layer == 3 && plan.n_qsa_layer == 1,
          "the plan counts three GDN blocks and one QSA block");
    check(plan.gdn_recurrent_bytes ==
              3ull * 48ull * 128ull * 128ull * sizeof(float),
          "the GDN recurrent state is 48 x 128 x 128 fp32 per GDN block");
    check(plan.qsa_kv_bytes == 2ull * n_ctx * 2ull * 256ull * sizeof(float),
          "the QSA cache is a key and a value plane per QSA block");
    /* The speculative cycle's per-row state slots are charged here too: one
     * slot per draft, each the size of the running state beside it.  They are
     * charged whether or not a drafter is armed, so the plan a serve reads
     * does not shrink under a serial leg and then overrun under an MTP one. */
    check(plan.spec_snapshot_slots == DS4_QWEN4EXP_IMPLEMENTED_DEPTH,
          "the plan holds one state slot per draft");
    check(plan.spec_snapshot_bytes ==
              (uint64_t)DS4_QWEN4EXP_IMPLEMENTED_DEPTH *
                  (plan.gdn_recurrent_bytes + plan.gdn_conv_bytes +
                   plan.ple_state_bytes),
          "each slot is the running state it mirrors");
    check(plan.total_bytes > 0 &&
              plan.total_bytes == plan.gdn_recurrent_bytes + plan.gdn_conv_bytes +
                                  plan.qsa_kv_bytes + plan.qsa_indexer_bytes +
                                  plan.ple_state_bytes +
                                  plan.spec_snapshot_bytes +
                                  plan.activation_bytes,
          "the session total is the sum of its families");

    printf("MEMORY SAFETY: the guard refuses before it allocates\n");
    check(refused, "one GiB free refuses the session");
    check(after_refusal == before_refusal,
          "the refused session allocated no device buffer");

    printf("RUN: the forward on the synthetic model\n");
    /* Sized from the writer's default vocabulary with room to spare; the run
     * hook reports the real one and the check below holds it to it. */
    float *logits = calloc((size_t)runs * vocab_cap, sizeof(float));
    /* n_hc * n_embd floats per token per run, at production per-layer shapes. */
    float *hyper = calloc(hyper_cap, sizeof(float));
    if (!logits || !hyper) {
        fprintf(stderr, "logit or hyper buffer allocation failed\n");
        return 2;
    }
    uint32_t hyper_row_floats = 0;
    int hyper_is_mixer_input = 0;
    /* 4 blocks x at most 7 slices, plus the embed, final mixer and head. */
    uint8_t trace[64];
    uint32_t trace_len = 0;

    const uint32_t completed = ds4_qwen4exp_test_graph_run(
            path, n_ctx, tokens, n_tokens, runs, plenty, logits, hyper,
            &hyper_row_floats, &hyper_is_mixer_input, trace, &trace_len,
            (uint32_t)sizeof(trace), &n_vocab);
    check(n_vocab > 0 && n_vocab <= vocab_cap,
          "the model reports a vocabulary this test can hold");
    check(completed == runs, "every forward completed");

    if (completed == runs && n_vocab > 0 && n_vocab <= vocab_cap) {
        printf("NUMBERS: the tower produces finite values, not NaN\n");
        /* The synthetic payload decodes to finite, small weights, so the
         * forward has to come out finite.  Without this the bit-exactness and
         * identity checks below ride on NaN, which compares equal to itself
         * bit for bit and hides a mutation that moves the numbers. */
        check(all_finite(logits, (size_t)runs * n_vocab),
              "every logit of every run is finite");
        check(all_finite(hyper, hyper_cap),
              "every pre-final-mixer row of every run is finite");
        check(any_nonzero(logits, (size_t)runs * n_vocab) &&
                  any_nonzero(hyper, hyper_cap),
              "the tower moved the numbers off zero");

        printf("RUN: a serial free run is bit-exact over three runs\n");
        const size_t row = (size_t)n_vocab * sizeof(float);
        check(memcmp(logits, logits + n_vocab, row) == 0,
              "run 2 is bit-identical to run 1");
        check(memcmp(logits, logits + 2u * n_vocab, row) == 0,
              "run 3 is bit-identical to run 1");

        printf("RUN: the pre-final-mixer rows the MTP head reads\n");
        check(hyper_row_floats == 4u * 2560u,
              "one pre-final-mixer row is n_hc x n_embd floats");
        /* Proven red: copying one row of s->mixed over s->hyper straight after
         * the final mixer -- what a mixer writing through its input would look
         * like from outside -- fails this check.  It did NOT fail before L7b
         * gave the synthetic checkpoint finite payloads, because every value
         * was NaN and memcmp compared identical NaN patterns.  Pointer identity
         * would not catch this mutant at all. */
        check(hyper_is_mixer_input == 1,
              "the rows the accessor returns are byte-for-byte what the final "
              "mixer read");
        const size_t hyper_run = (size_t)n_tokens * hyper_row_floats *
                                 sizeof(float);
        check(memcmp(hyper, hyper + (size_t)n_tokens * hyper_row_floats,
                     hyper_run) == 0,
              "run 2 pre-final-mixer rows are bit-identical to run 1");
        check(memcmp(hyper, hyper + 2u * (size_t)n_tokens * hyper_row_floats,
                     hyper_run) == 0,
              "run 3 pre-final-mixer rows are bit-identical to run 1");
    }
    printf("RUN: the slices ran in the model's order\n");
    /* The synthetic set is 4 blocks: GDN, GDN+PLE, GDN, QSA.  Per block the
     * order is PLE (block 1 only, FIRST), attention mixer, the block, attention
     * inject, FFN mixer, MoE, FFN inject -- then the final mixer and the head.
     * MLX Qwen4Exp.swift:113-121 is what puts PLE at the head of the layer. */
    static const uint8_t expected[] = {
        QW_SLICE_EMBED,
        /* blk.0, GDN */
        QW_SLICE_ATTN_MIX, QW_SLICE_GDN, QW_SLICE_ATTN_INJECT,
        QW_SLICE_FFN_MIX, QW_SLICE_MOE, QW_SLICE_FFN_INJECT,
        /* blk.1, GDN, carries the n-gram embedding */
        QW_SLICE_PLE,
        QW_SLICE_ATTN_MIX, QW_SLICE_GDN, QW_SLICE_ATTN_INJECT,
        QW_SLICE_FFN_MIX, QW_SLICE_MOE, QW_SLICE_FFN_INJECT,
        /* blk.2, GDN */
        QW_SLICE_ATTN_MIX, QW_SLICE_GDN, QW_SLICE_ATTN_INJECT,
        QW_SLICE_FFN_MIX, QW_SLICE_MOE, QW_SLICE_FFN_INJECT,
        /* blk.3, QSA */
        QW_SLICE_ATTN_MIX, QW_SLICE_QSA, QW_SLICE_ATTN_INJECT,
        QW_SLICE_FFN_MIX, QW_SLICE_MOE, QW_SLICE_FFN_INJECT,
        QW_SLICE_FINAL_MIX, QW_SLICE_HEAD,
    };
    const uint32_t expected_len = (uint32_t)(sizeof(expected) / sizeof(expected[0]));
    check(trace_len == expected_len, "the forward ran the expected slice count");
    if (trace_len == expected_len) {
        uint32_t first_bad = expected_len;
        for (uint32_t i = 0; i < expected_len; i++) {
            if (trace[i] != expected[i]) { first_bad = i; break; }
        }
        if (first_bad != expected_len) {
            printf("        first divergence at slice %u: ran %u, expected %u\n",
                   first_bad, trace[first_bad], expected[first_bad]);
        }
        /* Proven red: moving the PLE slot back to the tail of the layer --
         * the placement bug this assertion exists for -- diverges at slice 7,
         * running QW_SLICE_ATTN_MIX where QW_SLICE_PLE belongs. */
        check(first_bad == expected_len,
              "every slice ran in the model's order, PLE at the head of blk.1");
    }

    printf("SPANS: startup tensor-span preparation is per shard\n");
    {
        /* The CUDA startup pass copies tensor spans into device memory.  Its
         * copy half needs CUDA; its SHARD ARITHMETIC is what a split GGUF
         * breaks and is portable, so the eight-shard model covers it here.
         * Before this it refused split sets outright, which would have made the
         * four-shard production artifact take the refusal at every open. */
        uint64_t nspan = 0, bytes = 0, skipped = 0;
        uint32_t shards_seen = 0;
        int sorted_ok = 0, within_ok = 0, q8_ok = 0;
        uint32_t shard0_tensors = 99;
        const int rc = ds4_qwen4exp_test_collect_tensor_spans(
                path, &nspan, &bytes, &skipped, &shards_seen, &sorted_ok,
                &within_ok, &q8_ok, &shard0_tensors);
        check(rc == 0, "the span collector accepted the eight-shard model");
        if (rc == 0) {
            check(nspan > 0, "the collector produced spans");
            check(shards_seen > 1,
                  "spans came from more than one shard, so the split is really "
                  "being walked");
            check(within_ok == 1,
                  "every span fits inside the shard it names");
            check(sorted_ok == 1,
                  "spans are ordered by (shard, offset), so merging cannot join "
                  "two mappings");
            check(skipped > 0,
                  "the SSD-resident n-gram table was excluded from the spans");
            /* The production artifact's shard 0 is metadata only.  The fixture
             * matches it, which is what makes the old shard-0 bound reject
             * every tensor rather than merely misaddress some. */
            check(shard0_tensors == 0,
                  "shard 0 holds no tensors, as the production artifact's does "
                  "not");
            check(q8_ok == 1,
                  "every Q8_0 tensor ranges inside its own shard, so the "
                  "dequant cache can address it");
        }
    }

    printf("SHARDS: the same model in one file gives the same answer\n");
    {
        /* Same tensors, same name-seeded payloads, one file instead of three.
         * A weight resolved through a sibling's mapping diverges here and
         * nowhere else. */
        float *single = calloc(vocab_cap, sizeof(float));
        float *single_hyper = calloc(hyper_cap, sizeof(float));
        if (single && single_hyper) {
            uint32_t s_vocab = 0, s_row = 0;
            int s_mixer = 0;
            uint8_t s_trace[64];
            uint32_t s_trace_len = 0;
            const uint32_t s_runs = ds4_qwen4exp_test_graph_run(
                    single_path, n_ctx, tokens, n_tokens, 1, plenty, single,
                    single_hyper, &s_row, &s_mixer, s_trace, &s_trace_len,
                    (uint32_t)sizeof(s_trace), &s_vocab);
            check(s_runs == 1 && s_vocab == n_vocab,
                  "the one-shard model opens and runs");
            if (s_runs == 1 && s_vocab == n_vocab) {
                check(memcmp(single, logits,
                             (size_t)n_vocab * sizeof(float)) == 0,
                      "one-shard logits are bit-identical to the eight-shard run");
                check(s_row == hyper_row_floats &&
                          memcmp(single_hyper, hyper,
                                 (size_t)n_tokens * s_row * sizeof(float)) == 0,
                      "one-shard pre-final-mixer rows are bit-identical too");
                check(s_trace_len == trace_len &&
                          memcmp(s_trace, trace, s_trace_len) == 0,
                      "the one-shard run ran the same slices in the same order");
            }
        }
        free(single_hyper);
        free(single);
    }
    free(hyper);
    free(logits);
    }  /* !mixed_phase && !session_phase */

    if (session_phase) {
    printf("SESSION: the GLM graph is unreachable for a qwen4exp session\n");
    {
        /* qwen4exp shares the GLM_DSA family tag, so every ds4_session_is_glm()
         * branch in ds4.c is a hazard now that qwen4exp sessions exist: the GLM
         * graph reads ds4_weights, which a qwen4exp open never fills.  Open the
         * engine and a session the way a caller would and check what the
         * session came up as. */
        int is_glm = -1, glm_ready = -1, is_qw = -1, spec_routed = -1;
        int entry_refused = -1, entry_total = -1;
        const int rc = ds4_qwen4exp_test_session_glm_unreachable(
                single_path, &is_glm, &glm_ready, &is_qw, &spec_routed,
                &entry_refused, &entry_total);
        check(rc == 0, "the engine and a session open on a qwen4exp model");
        if (rc == 0) {
            check(is_qw == 1, "the session is recognised as qwen4exp");
            check(is_glm == 0,
                  "ds4_session_is_glm() is false, so the GLM graph is unreachable");
            check(glm_ready == 0, "no GLM graph was built for the qwen4exp session");
            check(spec_routed == 0,
                  "the speculative cycle is not routed with no MTP head loaded");
            /* Every public session entry qwen4exp does not serve, called for
             * real.  The COUNT is the point: an entry added to ds4.h without a
             * family guard shows up here as a shortfall rather than waiting for
             * someone to remember it. */
            printf("        %d of %d unsupported session entries refused\n",
                   entry_refused, entry_total);
            check(entry_total > 0 && entry_refused == entry_total,
                  "every unsupported session entry refuses by name");

            /* The list above is hand-written, so it cannot notice a NEW entry.
             * tests/qwen4exp_session_entries.sh derives the not-served set from
             * ds4.h itself; every name it prints must be one this test has
             * reviewed, either guarded or explicitly judged safe.  An entry
             * added to the header lands in neither and fails here. */
            static const char *const reviewed[] = {
                /* Guarded, and called for real above or by name in ds4.c. */
                "ds4_session_distributed_route_ready",
                "ds4_session_eval_layer_slice",
                "ds4_session_eval_output_head_from_hc",
                "ds4_session_gpu_warmup",
                "ds4_session_layer_payload_bytes",
                "ds4_session_layer_slice_reset",
                "ds4_session_load_layer_payload",
                "ds4_session_load_payload",
                "ds4_session_load_snapshot",
                "ds4_session_payload_bytes",
                "ds4_session_save_layer_payload",
                "ds4_session_save_payload",
                "ds4_session_save_snapshot",
                "ds4_session_set_directional_steering_ffn",
                "ds4_session_set_logits",
                "ds4_session_stage_payload",
                "ds4_session_sync_multimodal",
                "ds4_session_tp_spec_cycle",
                "ds4_sessions_eval_batch",
                "ds4_sessions_eval_batch_with_prefill",
                /* Safe without a guard: pure reads of fields a qwen4exp session
                 * sets truthfully, or of state it never enters. */
                "ds4_session_directional_steering_ffn",  /* reads the scale back */
                "ds4_session_has_vision_state",          /* false: no vision    */
                "ds4_session_is_distributed",            /* false: not routed   */
                "ds4_session_prefill_cap",               /* reads ctx sizing    */
                "ds4_session_rewrite_from_common",       /* rewind path, guarded */
                "ds4_session_rewrite_requires_rebuild",  /* rewind path, guarded */
                "ds4_session_vision_state_matches",      /* false: no vision    */
                "ds4_session_write_staged_payload",      /* takes a payload, not
                                                          * a session */
            };
            FILE *lp = popen("sh tests/qwen4exp_session_entries.sh", "r");
            check(lp != NULL, "the ds4.h entry list script ran");
            if (lp) {
                char name[128];
                int unreviewed = 0, listed = 0;
                while (fgets(name, sizeof(name), lp)) {
                    name[strcspn(name, "\r\n")] = '\0';
                    if (name[0] == '\0') continue;
                    listed++;
                    int seen = 0;
                    for (size_t k = 0;
                         k < sizeof(reviewed) / sizeof(reviewed[0]); k++) {
                        if (strcmp(name, reviewed[k]) == 0) { seen = 1; break; }
                    }
                    if (!seen) {
                        printf("        UNREVIEWED session entry: %s\n", name);
                        unreviewed++;
                    }
                }
                pclose(lp);
                printf("        %d not-served entries in ds4.h, %d unreviewed\n",
                       listed, unreviewed);
                check(listed > 0 && unreviewed == 0,
                      "every not-served entry in ds4.h has been reviewed");
            }
        }
    }

    printf("SERIAL SESSION: the public verbs on a qwen4exp session\n");
    {
        /* create -> sync -> argmax -> eval x32 -> top_logprobs -> invalidate ->
         * re-sync, all through ds4.h, on the model a caller would open.  This is
         * the path the CLI's serial leg takes. */
        enum { GEN = 32 };
        int first[3] = { -1, -1, -1 };
        int gen[3][GEN];
        int top_ok = 0, reset_first = -1, batch_refused = 0;
        int pos_sync = -1, pos_gen = -1, tape_len = -1, tape_ok = 0;
        int rc = 0;
        for (int run = 0; run < 3 && rc == 0; run++) {
            memset(gen[run], 0, sizeof(gen[run]));
            rc = ds4_qwen4exp_test_session_serial(
                    single_path, tokens, (int)n_tokens, GEN, &first[run],
                    gen[run], run == 0 ? &top_ok : NULL,
                    run == 0 ? &reset_first : NULL,
                    run == 0 ? &batch_refused : NULL,
                    run == 0 ? &pos_sync : NULL, run == 0 ? &pos_gen : NULL,
                    run == 0 ? &tape_len : NULL, run == 0 ? &tape_ok : NULL);
        }
        check(rc == 0, "the session ran sync, argmax, eval and invalidate");
        if (rc == 0) {
            check(first[0] >= 0 && first[0] < (int)n_vocab,
                  "the first argmax is a valid token id");
            check(memcmp(gen[0], gen[1], sizeof(gen[0])) == 0 &&
                      memcmp(gen[0], gen[2], sizeof(gen[0])) == 0,
                  "three serial runs generate the same 32 tokens");
            check(top_ok == 1,
                  "top_logprobs returns exactly 8 entries in descending order");
            check(reset_first == first[0],
                  "invalidate and re-sync reproduce the first token");
            check(batch_refused == 1,
                  "a batched call on a qwen4exp session refuses by name");
            /* ds4_session_pos() reads the checkpoint tape.  Unmaintained it
             * reports 0 forever, the generate loop's own bound goes wrong and
             * the server's prefix reuse compares against an empty tape. */
            check(pos_sync == (int)n_tokens,
                  "position after sync equals the prompt length");
            check(pos_gen == (int)n_tokens + GEN,
                  "position advances one per generated token");
            check(tape_len == (int)n_tokens + GEN,
                  "the token tape holds the prompt plus every generated token");
            check(tape_ok == 1,
                  "the tape's tail is the generated tokens, in order");
        }
    }

    printf("BATCH INVARIANCE: row 0 of a two-row forward against one row\n");
    {
        /* The cycle's correctness rests on this and nothing else asserted it.
         * No rollback is involved: the second state is rebuilt from the prompt,
         * so a difference here is the tower's, not the rewind's. */
        int a2 = -1, a1 = -1;
        double worst = -1.0, hc_worst = -1.0, content = -1.0, scale = -1.0;
        /* Widths 2..6, not just 2.  A prefill runs whatever width the caller
         * asked for, and row 0 of every one of them must equal the one-row
         * decode.  Two alone missed that widths of three and up did not. */
        double widest = -1.0, widest_hc = -1.0, widest_seq = -1.0;
        int wide_bad = 0;
        for (uint32_t w = 2u; w <= 6u; w++) {
            int wa2 = -1, wa1 = -1;
            double ww = -1.0, wh = -1.0, wc = -1.0, ws = -1.0, wseq = -1.0;
            double sb[6], sq[6];
            memset(sb, 0, sizeof(sb)); memset(sq, 0, sizeof(sq));
            const int wrc = ds4_qwen4exp_test_batch_invariance(
                    single_path, tokens, (int)n_tokens, tokens[0], w,
                    &wa2, &wa1, &ww, &wh, &wc, &ws, &wseq, sb, sq, 6);
            if (wrc != 0) { wide_bad = 1; continue; }
            printf("        width %u: row-0 diff %.6g, pre-mixer %.6g, "
                   "content %.6g, LAST-row vs sequential %.6g\n",
                   w, ww, wh, wc, wseq);
            {
                static const char *bn[6] = { "blk0","blk1","blk2","blk3",
                                             "ple_conv","ple_hist" };
                printf("          state after batched vs sequential:");
                int anyd = 0;
                for (int k = 0; k < 6; k++) {
                    if (sb[k] != sq[k]) { printf(" %s", bn[k]); anyd = 1; }
                }
                printf("%s\n", anyd ? "" : " identical");
            }
            if (ww > widest) widest = ww;
            if (wh > widest_hc) widest_hc = wh;
            if (wseq > widest_seq) widest_seq = wseq;
        }
        check(wide_bad == 0, "every batched width ran");
        /* Every width, not just the cycle's two.  Both row-count-tiered ops --
         * the Q8_0 matmul and ds4_gpu_matmul_f32_tensor -- now go through
         * decode-order entries at any width (ds4_qwen4exp_matmul.h), so a
         * forward of n rows gives each row exactly what a one-row decode would.
         * Before that, widths of three and up moved row 0 by 2e-3 and the last
         * row by 3e-3, and in a recurrent tower that compounds. */
        check(widest == 0.0 && widest_hc == 0.0,
              "row 0 is bit-identical to a one-row decode at every width");
        check(widest_seq == 0.0,
              "and so is the LAST row, against the same rows fed one at a "
              "time");

        const int rc = ds4_qwen4exp_test_batch_invariance(
                single_path, tokens, (int)n_tokens, tokens[0], 2u,
                &a2, &a1, &worst, &hc_worst, &content, &scale, NULL,
                NULL, NULL, 0);
        check(rc == 0, "the two-row and one-row forwards both ran");
        if (rc == 0) {
            printf("        two-row row 0 argmax %d, one-row argmax %d, "
                   "largest logit difference %.6g, largest "
                   "pre-final-mixer difference %.6g\n",
                   a2, a1, worst, hc_worst);
            printf("        row 0 against a DIFFERENT second row: %.6g "
                   "(row scale %.6g)\n", content, scale);
            check(content == 0.0,
                  "row 0 does not move when row 1 changes, so nothing leaks "
                  "backwards through the batch");
            check(a2 == a1,
                  "row 0 of a two-row forward picks the one-row token");
            /* BIT IDENTICAL, on both backends, and now asserted as such.
             *
             * Two ops in this tower chose a reduction strategy by row count:
             * the upstream Q8_0 matmul (ds4_metal.m:18193, against its
             * decode-order entry at :18263) and ds4_gpu_matmul_f32_tensor
             * (cuBLAS SGEMM above one row on CUDA, a matvec at one row on
             * Metal).  Both are routed inside the speculative cycle's width so
             * that row t of an n-row call IS a one-row call -- see
             * ds4_qwen4exp_matmul.h.  Every other op in the tower runs one
             * threadgroup or block per token and never sees the batch.
             *
             * With both routed the spread is 0 on Metal AND on CUDA, where it
             * was 2.37e-2 before.  So this is no longer a budget with headroom
             * in it: any nonzero value means an op has started tiering by row
             * count again, and that is exactly what should go red.
             *
             * It remains a TRIPWIRE and not the scoring contract -- depth 1 is
             * scored against the mtp1 oracle golden, see QWEN4EXP-PARITY.md --
             * but the tripwire can now be set where the truth is. */
            check(worst == 0.0,
                  "the batched verify's logits are bit-identical to the "
                  "one-row decode's");
            check(hc_worst == 0.0,
                  "and so is the pre-final-mixer row they come from");
        }
    }

    /* THE REAL VOCABULARY, before anything builds a prompt from it.
     *
     * This phase used `vocab_cap`, a compile-time upper bound of 65536, and
     * the chunking prompt below is built modulo it.  On this fixture the model
     * declares 4096, so those ids addressed rows the embedding table does not
     * have: an out-of-bounds read of the weight mapping on Metal, which does
     * not trap and so passed for months, and an illegal memory access on CUDA,
     * which does.  The chunking checks could never see it -- they compare
     * chunk widths with each other, and every width read the same wrong row. */
    {
        const uint32_t real_vocab = ds4_qwen4exp_test_model_vocab(single_path);
        check(real_vocab > 0, "the model reports its vocabulary");
        if (real_vocab > 0) {
            printf("        model vocabulary %u (was assuming %u)\n",
                   real_vocab, n_vocab);
            n_vocab = real_vocab;
        }
    }

    printf("MODEL REOPEN: a closed mapping leaves the backend nothing\n");
    {
        /* mmap hands the same address back on nearly every reopen in this
         * process, and both backends short-circuit their model registry when
         * the base and the size both match -- so a registry that outlives a
         * close is handed straight to the next mapping.  On CUDA that is a set
         * of device ranges and host registrations pointing into memory that
         * has been unmapped; on Metal a set of MTLBuffers built over it with
         * newBufferWithBytesNoCopy.  Forty cycles is enough for the address to
         * be recycled many times over. */
        int held = -1, same = -1;
        const int rrc = ds4_qwen4exp_test_model_reopen_registry(
                single_path, 40, &held, &same);
        check(rrc == 0, "forty open/close cycles ran");
        if (rrc == 0) {
            printf("        40 cycles: %d reopened at the previous address, "
                   "largest registry left after a close %d\n", same, held);
            check(held == 0,
                  "the backend holds no model-derived range after a close");
        }
    }

    printf("CLI FIRST TOKEN: the token the CLI would generate first is the "
           "argmax the adapter reports\n");
    {
        /* The g1 boot leg generated nothing while the same weights through the
         * protocol adapter generated coherent text, and the decoder carried
         * the suspicion for a day.  The decoder was never the difference: at
         * temperature 0 ds4_session_sample() returns sample_argmax(), the
         * function ds4_session_argmax() calls, so the two paths agree on the
         * token.  They disagree on what to DO with it -- the adapter passes
         * eos_token -1 and never stops, the CLI stops on the stop set -- and
         * they were not even being given the same prompt, because a golden
         * case is authored over an exact token count and the CLI sent the
         * whole file.  Hold the part that is ours: same state, same token. */
        const char *probe = "The quick brown fox jumps over the lazy dog. "
                            "It was a dark and stormy night and the wind "
                            "howled across the empty fields beyond the "
                            "village, mile after mile.";
        int full = -1, sent = -1, sampled = -1, argmax = -1;
        int is_stop = -1, is_stop_nothink = -1;
        const int frc = ds4_qwen4exp_test_cli_first_token(
                single_path, probe, 0, &full, &sent, &sampled, &argmax,
                &is_stop, &is_stop_nothink);
        check(frc == 0, "the CLI first-token probe ran");
        if (frc == 0) {
            printf("        prompt %d tokens, first sampled %d, argmax %d, "
                   "stop(think) %d, stop(no-think) %d\n",
                   sent, sampled, argmax, is_stop, is_stop_nothink);
            check(full == sent,
                  "an uncapped prompt sends every token it tokenized to");
            check(sampled == argmax,
                  "at temperature 0 the CLI's first token is the argmax the "
                  "adapter reports");
        }

        /* --prompt-tokens: the CLI can be handed the same PREFIX a golden case
         * was authored over.  Without it the CLI sends the whole file and the
         * two paths are compared on different text. */
        int cfull = -1, csent = -1, csampled = -1, cargmax = -1;
        const int cap = 16;
        const int crc2 = ds4_qwen4exp_test_cli_first_token(
                single_path, probe, cap, &cfull, &csent, &csampled, &cargmax,
                NULL, NULL);
        check(crc2 == 0, "the capped first-token probe ran");
        if (crc2 == 0 && frc == 0) {
            printf("        capped to %d of %d tokens, first sampled %d, "
                   "argmax %d\n", csent, cfull, csampled, cargmax);
            check(cfull == full,
                  "the cap does not change how the prompt tokenizes");
            check(csent == cap, "the cap sends exactly the tokens it names");
            check(csampled == cargmax,
                  "and the capped prompt's first token is its argmax too");
        }
    }

    printf("WIDTH AND CARRY: a wide prefill then three decodes, against the "
           "same rows one at a time\n");
    {
        /* The block above stops at width six because the hook it calls clamps
         * `wide_rows` to 2..8 and stages the tokens through an int32_t[8].
         * A real prefill runs at whatever width the caller asked for, up to
         * DS4_QWEN4EXP_MAX_PREFILL_ROWS, and the one row-count threshold left
         * in this tower -- ds4_gpu_glm53_matmul_bf16, ds4_metal.m:44625,
         * `use_mv = n_rows <= 8u`, which the QSA indexer's two BF16
         * projections go through -- sits at eight, exactly where the old
         * ceiling was.  Nine is the first width that takes the other kernel.
         *
         * A wide call that leaves different carried state does not have to
         * show it in that call's own logits: the state is what the NEXT
         * token reads.  So each width runs three single-row decodes after the
         * wide call and compares those too, and the carried state is compared
         * object by object rather than summed per layer.
         *
         * The context has to hold the prompt, the widest forward and the
         * three decodes. */
        /* Below eight the expert projections take the per-row kernels and the
         * result must equal a serial decode BIT FOR BIT: those are the widths
         * the speculative cycle runs, one row to decode and two to four to
         * verify at depths one to three.  At eight and above a prefill takes
         * the tensor-core tile, whose fold is one ascending accumulator where
         * the per-row kernel uses lane partials and a butterfly -- different
         * summation orders of the same products.  A prefill is identical
         * across depths by construction, so those widths are a TOLERANCE and
         * not an identity. */
        static const uint32_t widths[] =
            { 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 16u, 17u, 32u, 33u,
              64u, 128u, 512u, 1024u };
        enum { WC_DECODES = 3, WC_CTX = 4096 };
        /* The CUDA dense projections take an int8 tensor-core tile at eight
         * rows and up and the older exact kernels below that, so identity with
         * the one-at-a-time run is a requirement of the NARROW widths and a
         * measurement of the wide ones.  That line is where it is because the
         * speculative cycle is what needs the identity: it verifies at most
         * four rows, so every width it compares is narrow, while a prefill
         * chunk's width does not vary with draft depth and needs only to be
         * reproducible.  DS4_QWEN4EXP_NO_ROW_TILE puts every width back on the
         * exact kernels, and then the wide band must be zero as well. */
        const int wide_is_exact = getenv("DS4_QWEN4EXP_NO_ROW_TILE") != NULL;
        const uint32_t WC_EXACT_MAX = 7u;
        int wc_ran = 1;
        double wc_worst = 0.0;
        double wc_band = 0.0;
        int wc_state_bad = 0;
        int wc_state_wide = 0;
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
            const uint32_t w = widths[wi];
            double pre = -1.0, dec[WC_DECODES] = { -1.0, -1.0, -1.0 };
            double odiff = 0.0;
            char oname[256];
            int nobj = 0, ndiff = 0;
            oname[0] = '\0';
            const int wrc = ds4_qwen4exp_test_width_carry(
                    single_path, tokens, (int)n_tokens, tokens[0], w,
                    (uint32_t)WC_DECODES, (int)WC_CTX, &pre, dec,
                    oname, sizeof(oname), &odiff, &nobj, &ndiff);
            if (wrc != 0) {
                printf("        width %3u: hook returned %d\n", w, wrc);
                wc_ran = 0;
                continue;
            }
            printf("        width %3u: prefill last-row %.6g, decodes "
                   "%.6g / %.6g / %.6g, state objects %d differing of %d%s%s\n",
                   w, pre, dec[0], dec[1], dec[2], ndiff, nobj,
                   ndiff ? ", first " : "", ndiff ? oname : "");
            if (ndiff && (w <= WC_EXACT_MAX || wide_is_exact)) {
                printf("          objects that differ: %s (first gap %.6g)\n",
                       oname, odiff);
                /* The carried state is downstream of the logits, so it is
                 * exact exactly where they are: up to seven rows, and at every
                 * width when the tile is off. */
                if (w <= WC_EXACT_MAX || wide_is_exact) wc_state_bad = 1;
                else wc_state_wide = 1;
            }
            double worst_here = pre;
            for (int d = 0; d < WC_DECODES; d++) {
                if (dec[d] > worst_here) worst_here = dec[d];
            }
            if (w <= WC_EXACT_MAX || wide_is_exact) {
                if (worst_here > wc_worst) wc_worst = worst_here;
            } else if (worst_here > wc_band) {
                wc_band = worst_here;
            }
        }
        check(wc_ran == 1, "every width and carry case ran");
        check(wc_state_bad == 0,
              "up to seven rows a wide prefill leaves exactly the carried "
              "state the same rows leave one at a time, object by object");
        check(wc_worst == 0.0,
              "and up to seven rows the logits of the wide call and of the "
              "three decodes after it are bit-identical to the one-at-a-time "
              "run");
        printf("        eight rows and up, on the tensor-core tile: worst "
               "logit distance from the one-at-a-time run %.6g, carried state "
               "%s%s\n",
               wc_band, wc_state_wide ? "differs" : "identical",
               wide_is_exact ? " (tile off, so both are zero)" : "");
        /* The logits here are of order ten, so a relative 1e-2 is an absolute
         * 1e-1.  The synthetic set measured 0.046 on the tile (2026-09-04).
         * This is a ceiling on rounding, not a contract. */
        check(wc_band <= 1e-1,
              "at prefill widths the tile and the per-row path agree to "
              "rounding");
    }

    printf("PREFILL CHUNKING: the chunk boundary must not be observable\n");
    {
        /* A prompt synced in chunks of k must leave the same state as one
         * synced whole.  The batch-invariance probe above compares ROW 0 of a
         * two-row forward with a one-row forward; it says nothing about
         * whether row 1 advanced the caches, the recurrent state, the
         * convolution history and the n-gram history correctly.  A real boot
         * prefilled 1024 tokens in 512 chunks of two and produced one token
         * repeated, so this is the gap that mattered. */
        /* LONG ENOUGH TO CROSS THE BOUNDARIES.  Eight tokens exercise nothing:
         * the indexer pools its key tape in blocks (128 on the pinned shape),
         * the PLE convolution's dilated window is nine rows, and neither is
         * reached by a prompt shorter than they are.  300 tokens crosses the
         * first two pool blocks and fills the window several times over, and
         * the chunk widths below straddle those boundaries rather than
         * dividing them. */
        enum { LONG_LEN = 6, CHUNKS = 7 };
        static int longp[LONG_LEN];
        for (int i = 0; i < LONG_LEN; i++) {
            longp[i] = (int)((uint32_t)(i * 2654435761u + 12345u) %
                             (n_vocab > 1u ? n_vocab - 1u : 1u)) + 1;
        }
        static const uint32_t chunk[CHUNKS] =
            { 1u, 2u, 3u, 4u, 5u, 6u, 6u };
        int first[CHUNKS];
        int pos[CHUNKS];
        double worst_logit[CHUNKS];
        static double state[CHUNKS][6];
        static float ref_logits[64 * 1024];
        static float cmp_logits[64 * 1024];
        const int lcap = (int)(n_vocab < 64u * 1024u ? n_vocab : 64u * 1024u);
        int ran = 1;
        const int presync = (int)(getenv("DS4_QWEN4EXP_PRESYNC")
                                  ? atoi(getenv("DS4_QWEN4EXP_PRESYNC")) : 0);
        for (int c = 0; c < CHUNKS; c++) {
            pos[c] = -1;
            worst_logit[c] = 0.0;
            float *dst = (c == 0) ? ref_logits : cmp_logits;
            first[c] = ds4_qwen4exp_test_prefill_chunk_argmax(
                    single_path, longp, LONG_LEN, chunk[c], &pos[c],
                    dst, lcap, state[c], 6, 1, presync);
            if (first[c] < 0) ran = 0;
            if (c > 0 && first[c] >= 0) {
                for (int v = 0; v < lcap; v++) {
                    const double d = fabs((double)cmp_logits[v] -
                                          (double)ref_logits[v]);
                    if (d > worst_logit[c]) worst_logit[c] = d;
                }
            }
        }
        check(ran == 1, "every prefill chunking ran");
        if (ran) {
            printf("        argmax after a %d-token prompt by chunk:", LONG_LEN);
            for (int c = 0; c < CHUNKS; c++) {
                printf(" %u->%d", chunk[c], first[c]);
            }
            static const char *names[6] = {
                "blk0(GDN)", "blk1(GDN+PLE)", "blk2(GDN)", "blk3(QSA)",
                "ple_conv", "ple_hist" };
            printf("\n        carried state differing from chunk 1:");
            for (int c = 1; c < CHUNKS; c++) {
                printf(" [%u:", chunk[c]);
                int any = 0;
                for (int k = 0; k < 6; k++) {
                    if (state[c][k] != state[0][k]) {
                        printf(" %s", names[k]);
                        any = 1;
                    }
                }
                if (!any) printf(" none");
                printf("]");
            }
            printf("\n        largest logit difference against chunk 1:");
            for (int c = 1; c < CHUNKS; c++) {
                printf(" %u->%.3g", chunk[c], worst_logit[c]);
            }
            printf("\n");
            int same = 1, pos_ok = 1;
            for (int c = 1; c < CHUNKS; c++) {
                if (first[c] != first[0]) same = 0;
                if (pos[c] != pos[0]) pos_ok = 0;
            }
            /* REPORTED, not asserted.  The argmax DOES depend on the chunk,
             * because the per-forward residual above compounds through the
             * recurrent state.  Asserting it would restate the same open defect
             * as a second red line; the tripwire belongs on the cause. */
            if (!same) {
                printf("        NOTE: the argmax depends on the prefill chunk "
                       "-- the width residual above, compounded\n");
            }
            check(pos_ok && pos[0] == LONG_LEN,
                  "and every chunking leaves the position at the prompt "
                  "length");
        }
    }

    printf("SPECULATIVE SESSION: MTP legs at every depth against the serial "
           "leg\n");
    {
        /* The MTP leg has to produce the serial leg's tokens exactly -- the
         * cycle commits the target's own greedy argmax, so anything else is a
         * fault.  But the equality only MEANS that if rounds actually
         * speculated: a leg whose every round committed one token is the
         * serial leg wearing a different hat, and comparing it to itself is
         * vacuous.  So the acceptance counter is checked too, and the earlier
         * claim that the two legs matched was exactly this mistake.
         *
         * Every implemented depth runs, on the REAL graph rather than the
         * reference model: depth 3 verifies four rows and replays a partial
         * accept at two or three, so the widths the cycle depends on are
         * exercised against the tower that has to be invariant across them.
         */
        enum { GEN = 32 };
        char head[4096];
        snprintf(head, sizeof(head), "%s/qw4x-mtp.gguf",
                 argc >= 4 ? argv[3] : argv[2]);

        int serial[GEN];
        int sfirst = -1;
        memset(serial, 0, sizeof(serial));
        const int src = ds4_qwen4exp_test_session_serial(
                single_path, tokens, (int)n_tokens, GEN, &sfirst, serial,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        check(src == 0, "the serial leg ran to 32 tokens");

        for (int dt = 2; src == 0 && dt <= DS4_QWEN4EXP_MTP_MAX_COMMIT; dt++) {
            const int depth = dt - 1;
            printf("      depth %d\n", depth);

            int spec[GEN];
            int n_spec = 0, pos_gen = -1, tape_len = -1, tape_ok = 0;
            int rounds = -1, accepted = -1, deep = -1, counters_ok = 0;
            int disagree = -1;
            memset(spec, 0, sizeof(spec));
            /* Leg 1: the REAL head.  Its drafts are noise, so this exercises
             * the reject-and-replay half -- which is where the rounds were
             * failing, and which at depth 2 and up also has to unwind a
             * speculative head-cache chain. */
            const int rc = ds4_qwen4exp_test_session_spec(
                    single_path, head, tokens, (int)n_tokens, GEN, dt,
                    NULL, 0,
                    spec, &n_spec, &pos_gen, &tape_len, &tape_ok,
                    &rounds, &accepted, &deep, &counters_ok, &disagree,
                    NULL, NULL, NULL, NULL);

            if (rc != 0) printf("        speculative rc %d\n", rc);
            check(rc == 0, "the speculative leg ran to 32 tokens");
            if (rc != 0) continue;

            /* The disagreement counter on the REAL tower.  Every round here
             * rejects, so this is the batch-shape residual's runtime rate.
             * Exactness at every width the cycle uses is the contract, so a
             * non-zero count is a failure and not a budget. */
            printf("        %d tokens, %d rounds, %d accepted, "
                   "%d full commits, %d verify/replay disagreements\n",
                   n_spec, rounds, accepted, deep, disagree);
            check(n_spec == GEN, "the speculative leg emitted 32 tokens");
            check(counters_ok == 1, "the cycle's counters are self-consistent");
            check(disagree == 0,
                  "the wide verify and the narrow replay agree at every width");
            int same = (n_spec == GEN);
            int first_diff = -1;
            for (int i = 0; i < GEN && same; i++) {
                if (spec[i] != serial[i]) { same = 0; first_diff = i; }
            }
            check(same, "the MTP leg is the serial leg token for token");
            if (!same && first_diff >= 0) {
                printf("        first difference at %d: serial %d, MTP %d\n",
                       first_diff, serial[first_diff], spec[first_diff]);
            }
            /* The checkpoint after a COMPLETED round, which could not be
               asserted while no round completed. */
            check(pos_gen == (int)n_tokens + GEN,
                  "position after the MTP leg is the prompt plus 32");
            check(tape_len == (int)n_tokens + GEN,
                  "the tape holds the prompt plus every committed token");
            check(tape_ok == 1,
                  "the tape's tail is the committed tokens, in order");

            /* Leg 2: the ACCEPTING half.  The fixture's head is random, so it
             * drafts the right token about once in n_vocab tries and leg 1
             * above accepted nothing at all -- which makes its "equals the
             * serial leg" result true and empty.  Feed the drafts the serial
             * leg proves are right, and the wide verify, the borrowed per-row
             * head and the full-depth commit all have to run.
             *
             * The forced hook answers by ABSOLUTE position -- the draft for
             * pos + 2 -- so one array serves every depth: chain step k asks
             * for the token k + 1 past the frontier and gets it. */
            int forced[64];
            const int flen = (int)n_tokens + GEN;
            for (int i = 0; i < (int)n_tokens; i++) forced[i] = tokens[i];
            for (int i = 0; i < GEN; i++) forced[(int)n_tokens + i] = serial[i];

            int spec2[GEN];
            int n2 = 0, pos2 = -1, tape2 = -1, tail2 = 0;
            int rounds2 = -1, acc2 = -1, deep2 = -1, cok2 = 0;
            memset(spec2, 0, sizeof(spec2));
            int had = -1, inv = -1, syn = -1, evl = -1, dis2 = -1;
            const int rc2 = ds4_qwen4exp_test_session_spec(
                    single_path, head, tokens, (int)n_tokens, GEN, dt,
                    forced, flen,
                    spec2, &n2, &pos2, &tape2, &tail2,
                    &rounds2, &acc2, &deep2, &cok2, &dis2,
                    &had, &inv, &syn, &evl);
            check(rc2 == 0, "the leg ran with the drafts the serial leg proves");
            if (rc2 != 0) continue;

            printf("        %d tokens, %d rounds, %d accepted, "
                   "%d full commits, %d verify/replay disagreements\n",
                   n2, rounds2, acc2, deep2, dis2);
            /* NON-VACUITY.  Without this the comparison below is a leg that
             * committed one token per round being compared with the serial
             * leg it already is.  At depth N the bar is a FULL commit: a
             * depth-3 leg whose deepest round committed two tokens never ran
             * the three-draft accept. */
            check(deep2 > 0,
                  "rounds committed the fed token plus the whole chain, so the "
                  "full-depth accept path ran");
            check(rounds2 < n2,
                  "fewer rounds than tokens, so speculation saved work");
            check(dis2 == 0,
                  "the accepting leg had no verify/replay disagreement");
            int same2 = (n2 == GEN);
            for (int i = 0; i < GEN && same2; i++) {
                if (spec2[i] != serial[i]) same2 = 0;
            }
            check(same2,
                  "the accepting leg is the serial leg token for token");
            check(pos2 == (int)n_tokens + GEN && tape2 == pos2 && tail2 == 1,
                  "the checkpoint is truthful after accepting rounds");
            check(cok2 == 1, "its counters are self-consistent");

            /* The carried chain across a phase boundary.  `had` is the
             * non-vacuity guard: without a chain in hand the three checks
             * below would pass on a state that never carried one.  It reports
             * the chain LENGTH, so a verb that dropped only part of it is
             * still visible. */
            check(had == depth,
                  "the leg was carrying the whole chain to begin with");
            check(inv == 0, "invalidate drops the carried chain");
            check(syn == 0, "a sync drops the carried chain");
            check(evl == 0, "a serial eval drops the carried chain");
        }
    }

    printf("SUPPORT MODEL: the family gate at open, both directions\n");
    {
        /* A qwen4exp head on a qwen4exp target must be ADMITTED: the branch
         * that binds it was unreachable until the GLM-wide --mtp-model refusal
         * learned about the variant.  Anything else must be REFUSED, because
         * the other binders fill weight tables this graph never reads. */
        char head[4096];
        snprintf(head, sizeof(head), "%s/qw4x-mtp.gguf",
                 argc >= 4 ? argv[3] : argv[2]);
        int is_nextn = -1;
        const int admitted = ds4_qwen4exp_test_open_with_support(
                argv[0], single_path, head, &is_nextn);
        check(admitted == 0 && is_nextn == 1,
              "a qwen4exp nextn head is admitted on a qwen4exp target");

        /* The target itself as the support model: a qwen4exp file with no
         * nextn_predict_layers, so it detects as some other kind. */
        int other_nextn = -1;
        const int refused = ds4_qwen4exp_test_open_with_support(
                argv[0], single_path, single_path, &other_nextn);
        check(refused == 1,
              "a support model that is not a qwen4exp nextn head is refused");
    }

    printf("HEAD BLOCK: the MTP head's 49th block runs on a real head file\n");
    {
        /* The cycle test links no backend, so the block's only real coverage is
         * here: two live mappings, the head's own cache slot at layer 48, and
         * the same QSA/MoE/mixer functions the tower runs. */
        char head_path[4096];
        snprintf(head_path, sizeof(head_path), "%s/qw4x-mtp.gguf",
                 argc >= 4 ? argv[3] : argv[2]);
        int ran = 0, det = 0, changed = 0;
        const int rc = ds4_qwen4exp_test_head_block_run(
                single_path, head_path, n_tokens, &ran, &det, &changed);

        /* The Q8_0 inject weight decodes now, so the block runs end to end on
         * a real head file: two live mappings, the head's own cache slot at
         * layer 48, and the same QSA/MoE/mixer functions the tower runs. */
        check(rc == 0 && ran == 1, "the head block ran on the head GGUF");
        if (rc == 0 && ran == 1) {
            check(det == 1, "two runs of the head block are bit-identical");
            check(changed == 1,
                  "the head block changed the stream, so it did something");
        }
    }

    printf("HEAD LM HEAD: the head borrows the target's, bit for bit\n");
    {
        /* The head runs the borrowed LM head through its matmul hook, and the
         * hook is where a row-count-tiering entry could get back in.  Four rows
         * because that is the width the head's eh_proj asks for at one token,
         * and it was the width that was wrong. */
        int one_row = 0, invariant = 0;
        const int rc = ds4_qwen4exp_test_head_lm_head_identity(
                single_path, 4u, &one_row, &invariant);
        check(rc == 0, "the head's matmul hook ran on the target's LM head");
        if (rc == 0) {
            check(one_row == 1,
                  "the head's LM head is the tower's, bit for bit, at one row");
            check(invariant == 1,
                  "row t of the head's n-row call is its own one-row call");
        }
    }

    }  /* session_phase */

    if (mixed_phase) {
    printf("MIXED QUANT: a routed gate/up slab that is not Q4_K\n");
    /* Same model with one block's routed gate and up at Q5_K, the way the
     * shipped recipes mix types per block.  The loader binds it and the MoE
     * kernel decodes it, so the forward must RUN.  The numbers are
     * tests/test_qwen4exp_moe's job: this set leaves the expert slabs as
     * holes, and it is the dispatch that is under test here. */
    float *scratch = calloc(vocab_cap, sizeof(float));
    if (scratch) {
        uint32_t mixed_vocab = 0;
        const uint32_t mixed_runs = ds4_qwen4exp_test_graph_run(
                mixed_path, n_ctx, tokens, n_tokens, 1, plenty, scratch,
                NULL, NULL, NULL, NULL, NULL, 0, &mixed_vocab);
        check(mixed_runs == 1,
              "a Q5_K routed gate/up slab runs, it is not refused");
        free(scratch);
    }

    }  /* mixed_phase */

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
