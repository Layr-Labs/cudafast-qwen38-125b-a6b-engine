/*
 * qwen4exp loader tests.
 *
 * Drives the L1 loader against synthetic GGUF split sets written by
 * tools/qwen4exp_synthetic_gguf.py.  tests/test_qwen4exp_loader.sh builds the
 * good file and the fault-injected ones, then passes their directory here.
 *
 * Compiles only when ds4.c is built with -DDS4_TEST_HOOKS (the test target
 * adds this flag), matching the tests/test_engine_mgpu_placement.c pattern.
 * Every refusal case runs in a forked child because the loader refuses the way
 * the rest of ds4 does: a message on stderr and exit(1).
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

/* These match the definitions in ds4_qwen4exp.h, which cannot be included here
 * because it names ds4.c-internal types. */
#define DS4_QWEN4EXP_MEM_COUNT 8
enum {
    MEM_EXPERTS = 0, MEM_PLE, MEM_DENSE, MEM_EMBED,
    MEM_HC, MEM_GDN, MEM_ROUTER, MEM_INDEXER,
};

typedef struct {
    uint64_t bytes[DS4_QWEN4EXP_MEM_COUNT];
    uint64_t tensors[DS4_QWEN4EXP_MEM_COUNT];
    uint64_t total_bytes;
    uint64_t resident_bytes;
    uint64_t ssd_bytes;
    uint64_t bound_tensors;
} mem_plan;

/* Declared in ds4_qwen4exp.inc under DS4_TEST_HOOKS. */
void ds4_qwen4exp_test_open(const char *path, mem_plan *plan_out,
                            uint64_t *ple_rows_out, uint64_t *ple_row_bytes_out,
                            uint32_t *n_shards_out, uint64_t *n_tensors_out);
void ds4_qwen4exp_test_header_bytes(const char *path, uint64_t *total_out,
                                    uint64_t *ple_out);
uint64_t ds4_qwen4exp_test_shard_order(const char *path, uint32_t *out,
                                       uint64_t cap);
void ds4_qwen4exp_test_set_free_memory_override(uint64_t bytes, bool enable);
void ds4_qwen4exp_test_open_mtp(const char *path, uint64_t *bound_out,
                                uint64_t *n_tensors_out, uint64_t *bytes_out);

static int g_failures;
static int g_checks;

static void check(bool ok, const char *what) {
    g_checks++;
    if (ok) {
        printf("  ok    %s\n", what);
    } else {
        g_failures++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_eq_u64(uint64_t got, uint64_t want, const char *what) {
    g_checks++;
    if (got == want) {
        printf("  ok    %s (%llu)\n", what, (unsigned long long)got);
    } else {
        g_failures++;
        printf("  FAIL  %s: got %llu, want %llu\n", what,
               (unsigned long long)got, (unsigned long long)want);
    }
}

static uint64_t rss_bytes(void) {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t)info.resident_size;
#elif defined(__linux__)
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp) return 0;
    unsigned long long total = 0, resident = 0;
    if (fscanf(fp, "%llu %llu", &total, &resident) != 2) resident = 0;
    fclose(fp);
    return (uint64_t)resident * (uint64_t)sysconf(_SC_PAGESIZE);
#else
    return 0;
#endif
}

/*
 * Run one loader open in a child and report whether it refused, plus what it
 * said.  free_override < 0 means "leave the real probe alone".
 */
static bool run_child(const char *path, const char *expect_substring,
                      bool expect_refusal, long long free_override,
                      const char *what) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return false; }

    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); return false; }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDERR_FILENO);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        if (free_override >= 0) {
            ds4_qwen4exp_test_set_free_memory_override(
                    (uint64_t)free_override, true);
        }
        ds4_qwen4exp_test_open(path, NULL, NULL, NULL, NULL, NULL);
        /* _exit skips stdio flushing and the plan goes to a pipe. */
        fflush(NULL);
        _exit(0);
    }

    close(pipefd[1]);
    char buf[65536];
    size_t used = 0;
    ssize_t n;
    while (used + 1 < sizeof(buf) &&
           (n = read(pipefd[0], buf + used, sizeof(buf) - used - 1)) > 0) {
        used += (size_t)n;
    }
    buf[used] = '\0';
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    const bool exited = WIFEXITED(status);
    const int code = exited ? WEXITSTATUS(status) : -1;
    const bool refused = exited && code == 1;

    bool ok = (refused == expect_refusal);
    if (ok && expect_substring) ok = strstr(buf, expect_substring) != NULL;

    g_checks++;
    if (ok) {
        printf("  ok    %s\n", what);
    } else {
        g_failures++;
        printf("  FAIL  %s: exit=%d refused=%d expected_refusal=%d\n",
               what, code, (int)refused, (int)expect_refusal);
        if (expect_substring) printf("        wanted substring: %s\n", expect_substring);
        printf("        output was:\n%s\n", buf);
    }
    return ok;
}

static void join(char *dst, size_t size, const char *dir, const char *rel) {
    const int n = snprintf(dst, size, "%s/%s", dir, rel);
    if (n < 0 || (size_t)n >= size) {
        fprintf(stderr, "path too long: %s/%s\n", dir, rel);
        exit(2);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <dir built by tests/test_qwen4exp_loader.sh>\n",
                argv[0]);
        return 2;
    }
    /* --open PATH: bind one existing qwen4exp GGUF and print its plan, with the
     * memory guard told there is plenty free.  Used to run the loader against
     * the real artifact's metadata on a machine that cannot hold the weights. */
    if (strcmp(argv[1], "--open") == 0) {
        if (argc < 3) {
            fprintf(stderr, "usage: %s --open <shard 1 of a qwen4exp GGUF>\n",
                    argv[0]);
            return 2;
        }
        mem_plan p2;
        uint64_t r = 0, rb = 0, nt = 0;
        uint32_t ns = 0;
        memset(&p2, 0, sizeof(p2));
        ds4_qwen4exp_test_set_free_memory_override(1ull << 50, true);
        ds4_qwen4exp_test_open(argv[2], &p2, &r, &rb, &ns, &nt);
        printf("bound %llu of %llu tensors over %u shards\n",
               (unsigned long long)p2.bound_tensors,
               (unsigned long long)nt, ns);
        printf("PLE %llu rows x %llu B\n",
               (unsigned long long)r, (unsigned long long)rb);
        return p2.bound_tensors == nt ? 0 : 1;
    }

    if (strcmp(argv[1], "--open-mtp") == 0) {
        if (argc < 3) {
            fprintf(stderr, "usage: %s --open-mtp <MTP head GGUF>\n", argv[0]);
            return 2;
        }
        uint64_t bound = 0, nt = 0, bytes = 0;
        ds4_qwen4exp_test_open_mtp(argv[2], &bound, &nt, &bytes);
        printf("MTP head: bound %llu of %llu tensors, %.2f GiB\n",
               (unsigned long long)bound, (unsigned long long)nt,
               (double)bytes / (1024.0 * 1024.0 * 1024.0));
        return bound == nt ? 0 : 1;
    }

    /* --open-pair TARGET HEAD: the engine's own order.  The target open fixes
     * the geometry, then the head is validated and bound against it.  A head
     * that disagrees on any shared width is refused here, which is the check
     * that matters: the head borrows the target's token_embd and output. */
    if (strcmp(argv[1], "--open-pair") == 0) {
        if (argc < 4) {
            fprintf(stderr,
                    "usage: %s --open-pair <target shard 1> <MTP head GGUF>\n",
                    argv[0]);
            return 2;
        }
        mem_plan p2;
        uint64_t r = 0, rb = 0, nt = 0, bound = 0, hnt = 0, bytes = 0;
        uint32_t ns = 0;
        memset(&p2, 0, sizeof(p2));
        ds4_qwen4exp_test_set_free_memory_override(1ull << 50, true);
        ds4_qwen4exp_test_open(argv[2], &p2, &r, &rb, &ns, &nt);
        ds4_qwen4exp_test_open_mtp(argv[3], &bound, &hnt, &bytes);
        printf("target: bound %llu of %llu tensors over %u shards\n",
               (unsigned long long)p2.bound_tensors,
               (unsigned long long)nt, ns);
        printf("MTP head: bound %llu of %llu tensors, %.2f GiB\n",
               (unsigned long long)bound, (unsigned long long)hnt,
               (double)bytes / (1024.0 * 1024.0 * 1024.0));
        return (p2.bound_tensors == nt && bound == hnt && bound > 0) ? 0 : 1;
    }

    const char *dir = argv[1];
    const char *shard1 = "qw4x-00001-of-00003.gguf";
    char good[4096];
    join(good, sizeof(good), dir, "good/qw4x-00001-of-00003.gguf");

    /* Geometry the writer used; keep in step with test_qwen4exp_loader.sh. */
    const uint32_t layers = 4;
    /* The head vocabularies are 16 consecutive primes and the table height is
     * their sum, so it is not the round --ple-rows target: the writer rounds
     * 200000 up to the 16 primes at or above 12500. */
    const uint64_t ple_rows = 201000;
    const uint32_t shards = 3;

    printf("qwen4exp loader tests\n");

    /* ---------------------------------------------------------------- */
    printf("BIND: every tensor of the synthetic file\n");
    mem_plan plan;
    uint64_t rows = 0, row_bytes = 0, n_tensors = 0;
    uint32_t n_shards = 0;
    memset(&plan, 0, sizeof(plan));

    /* The binding and plan checks are about what the loader reads, not about
     * this laptop's free memory, so report plenty here.  The guard itself is
     * exercised on its own below, with the real refusal thresholds. */
    ds4_qwen4exp_test_set_free_memory_override(1ull << 40, true);
    const uint64_t rss_before = rss_bytes();
    ds4_qwen4exp_test_open(good, &plan, &rows, &row_bytes, &n_shards, &n_tensors);
    const uint64_t rss_after = rss_bytes();
    ds4_qwen4exp_test_set_free_memory_override(0, false);

    check_eq_u64(plan.bound_tensors, n_tensors,
                 "every tensor in the file is bound");
    check_eq_u64(n_shards, shards, "shard count");

    /* 3 GDN blocks and 1 QSA block at interval 4. */
    const uint64_t gdn_blocks = 3, qsa_blocks = 1;
    check_eq_u64(plan.tensors[MEM_EXPERTS], 3ull * layers, "expert tensors");
    check_eq_u64(plan.tensors[MEM_ROUTER], 2ull * layers, "router tensors");
    check_eq_u64(plan.tensors[MEM_GDN], 6ull * gdn_blocks, "GDN parameter tensors");
    check_eq_u64(plan.tensors[MEM_INDEXER], 4ull * qsa_blocks, "indexer tensors");
    check_eq_u64(plan.tensors[MEM_HC], 8ull * layers + 3ull,
                 "HC tensors including the output head");
    check_eq_u64(plan.tensors[MEM_EMBED], 2, "embed/output tensors");
    check_eq_u64(plan.tensors[MEM_PLE], 1, "PLE table tensors");
    check_eq_u64(plan.tensors[MEM_DENSE],
                 3ull * gdn_blocks    /* attn_qkv, attn_gate, ssm_out    */
                 + 6ull * qsa_blocks  /* q,k,v,output + q_norm, k_norm   */
                 + 3ull * layers      /* shared expert gate, up, down    */
                 + 6ull,              /* the PLE block projections       */
                 "dense tensors");

    /* ---------------------------------------------------------------- */
    printf("PLE: bound as an mmap handle\n");
    check_eq_u64(rows, ple_rows, "PLE table rows");
    check_eq_u64(row_bytes, 90, "IQ4_NL bytes per 160-value row");
    check_eq_u64(plan.ssd_bytes, rows * row_bytes, "PLE bytes are rows x stride");

    /* ---------------------------------------------------------------- */
    printf("MEMORY: no double residency\n");
    uint64_t header_total = 0, header_ple = 0;
    ds4_qwen4exp_test_header_bytes(good, &header_total, &header_ple);
    const uint64_t rss_growth = rss_after > rss_before ? rss_after - rss_before : 0;
    const uint64_t limit = 128ull * 1024ull * 1024ull;
    printf("        mapped %.2f GiB, RSS grew %.1f MiB\n",
           (double)header_total / (1024.0 * 1024.0 * 1024.0),
           (double)rss_growth / (1024.0 * 1024.0));
    check(rss_growth < limit,
          "RSS after open stays under 128 MiB (mmap resident only)");
    check(header_total > 4ull * 1024ull * 1024ull * 1024ull,
          "the test file really is multi-GiB, so the RSS bound means something");

    /* ---------------------------------------------------------------- */
    printf("PLAN: numbers equal the GGUF header byte counts\n");
    check_eq_u64(plan.total_bytes, header_total,
                 "plan total equals the sum of tensor byte counts");
    check_eq_u64(plan.ssd_bytes, header_ple, "plan PLE bytes equal the header");
    check_eq_u64(plan.resident_bytes, header_total - header_ple,
                 "resident equals total minus the SSD-resident table");

    /* ---------------------------------------------------------------- */
    printf("SHARD: join order\n");
    static uint32_t order[8192];
    const uint64_t total = ds4_qwen4exp_test_shard_order(good, order,
                                                         sizeof(order) / sizeof(order[0]));
    check_eq_u64(total, n_tensors, "shard join sees every tensor");
    bool monotonic = true;
    for (uint64_t i = 1; i < total && i < sizeof(order) / sizeof(order[0]); i++) {
        if (order[i] < order[i - 1]) monotonic = false;
    }
    check(monotonic, "tensors appear in shard order, shard 0 directory first");
    check(total > 0 && order[0] == 1,
          "shard 0 is metadata only, so the first tensor comes from shard 1");
    check(total > 0 && order[total - 1] == shards - 1,
          "the last tensor comes from the last shard");

    /* ---------------------------------------------------------------- */
    printf("REFUSE: a bad file is refused by name\n");
    char p[4096];

    join(p, sizeof(p), dir, "missing/"); strcat(p, shard1);
    run_child(p, "required tensor is missing: blk.0.ssm_a", true, -1,
              "missing tensor");

    join(p, sizeof(p), dir, "badtype/"); strcat(p, shard1);
    run_child(p, "blk.0.attn_qkv.weight has type q4_k, expected q8_0", true, -1,
              "mis-typed tensor");

    join(p, sizeof(p), dir, "badshape/"); strcat(p, shard1);
    run_child(p, "blk.0.attn_gate.weight has dim[1]=6100, expected 6144", true, -1,
              "mis-shaped tensor");

    join(p, sizeof(p), dir, "badple/"); strcat(p, shard1);
    run_child(p, "per_layer_token_embd.weight has dim[0]=128, expected 160",
              true, -1, "mis-shaped PLE table");

    /* ---------------------------------------------------------------- */
    printf("REFUSE: metadata that disagrees with the fixture geometry\n");

    join(p, sizeof(p), dir, "badkv_hc/"); strcat(p, shard1);
    run_child(p, "expected hyper_connection.low_rank=320", true, -1,
              "hyper_connection.low_rank disagrees with the fixture");

    join(p, sizeof(p), dir, "badkv_experts/"); strcat(p, shard1);
    run_child(p, "expected expert_count=512", true, -1,
              "expert_count disagrees with the fixture");

    join(p, sizeof(p), dir, "badkv_heads/"); strcat(p, shard1);
    run_child(p, "expected attention.head_count=24", true, -1,
              "attention.head_count disagrees with the fixture");

    join(p, sizeof(p), dir, "badkv_interval/"); strcat(p, shard1);
    run_child(p, "expected full_attention_interval=4", true, -1,
              "full_attention_interval disagrees with the fixture");

    join(p, sizeof(p), dir, "badkv_ple_onebased/"); strcat(p, shard1);
    run_child(p, "layer_ids_one_based", true, -1,
              "one-based and zero-based PLE block ids disagree");

    join(p, sizeof(p), dir, "badkv_ple_rows/"); strcat(p, shard1);
    run_child(p, "PLE head vocabularies sum to", true, -1,
              "PLE head vocabularies overrun the table height");

    join(p, sizeof(p), dir, "badkv_schedule/"); strcat(p, shard1);
    run_child(p, "compress ratio", true, -1,
              "the attention schedule disagrees with the interval rule");

    /* ---------------------------------------------------------------- */
    printf("MEMORY SAFETY: refuse when free memory cannot hold the model\n");
    const uint64_t headroom = 10ull * 1024ull * 1024ull * 1024ull;

    /* Negative control: a low reading must refuse. */
    run_child(good, "refusing to load: free unified memory", true,
              (long long)(1ull * 1024ull * 1024ull * 1024ull),
              "1 GiB free refuses the load");

    /* Just under the requirement still refuses. */
    run_child(good, "refusing to load: free unified memory", true,
              (long long)(plan.resident_bytes + headroom - 1ull),
              "one byte short of resident + 10 GiB refuses");

    /* Positive control: exactly enough loads. */
    run_child(good, "memory budget", false,
              (long long)(plan.resident_bytes + headroom),
              "resident + 10 GiB exactly is accepted");

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
