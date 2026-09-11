/*
 * qwen4exp -- Qwen3.8-Flash-Next 125B-A6B: metadata validation, tensor binding
 * and the open-time memory plan.  Included from ds4.c; see ds4_qwen4exp.h for
 * the surface L3-L7 build on.
 *
 * Two artifacts are known to carry general.architecture=qwen4exp and they spell
 * some keys differently, so every lookup that differs takes a small alias list
 * in the style of model_get_u32_any():
 *
 *   unsloth/Qwen3.8-Flash-Next-GGUF          UD-Q4_K_XL, 4 shards, 1224 tensors
 *   Baekpica/...-Mixed-Quant-SSD-PLE-GGUF    MQ-Q5-SSD-PLE-BF16, 3 shards
 *
 * Where they disagree the loader refuses by name rather than picking a winner.
 */

ds4_qwen4exp_config g_ds4_qwen4exp;

/* ------------------------------------------------------------------------
 * Free memory probe
 * ------------------------------------------------------------------------ */

#ifdef DS4_TEST_HOOKS
static bool     g_qwen4exp_free_memory_override_set;
static uint64_t g_qwen4exp_free_memory_override;

/* Test-only.  Production builds do not compile this, so there is no way to
 * talk the memory guard out of a refusal at runtime. */
void ds4_qwen4exp_test_set_free_memory_override(uint64_t bytes, bool enable) {
    g_qwen4exp_free_memory_override = bytes;
    g_qwen4exp_free_memory_override_set = enable;
}
#endif

/* Free unified memory in bytes, or 0 when the platform does not say.
 * Linux reads MemAvailable, which already accounts for reclaimable cache. */
static uint64_t qwen4exp_free_memory_bytes(void) {
#ifdef DS4_TEST_HOOKS
    if (g_qwen4exp_free_memory_override_set) {
        return g_qwen4exp_free_memory_override;
    }
    /* TEST BUILDS ONLY.  A production ds4 does not compile this, so there is
     * no way to talk the guard out of a refusal in the field -- which is the
     * point of the guard.  A hooks build reads it so the CLI can be exercised
     * end to end on a machine that cannot spare the real headroom. */
    {
        const char *env = getenv("DS4_QWEN4EXP_TEST_FREE_MEMORY_GIB");
        if (env && env[0]) {
            const double gib = atof(env);
            if (gib > 0.0) return (uint64_t)(gib * 1073741824.0);
        }
    }
#endif
#if defined(__linux__)
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t available = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            available = kib > UINT64_MAX / 1024ull ?
                UINT64_MAX : (uint64_t)kib * 1024ull;
            break;
        }
    }
    fclose(fp);
    return available;
#elif defined(__APPLE__)
    /* Darwin has no MemAvailable.  Free plus the reclaimable pages is the
     * closest equivalent for a unified-memory budget. */
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS) return 0;
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vmstat, &count) != KERN_SUCCESS) {
        return 0;
    }
    const uint64_t pages = (uint64_t)vmstat.free_count +
                           (uint64_t)vmstat.inactive_count +
                           (uint64_t)vmstat.purgeable_count +
                           (uint64_t)vmstat.external_page_count;
    return pages * (uint64_t)page_size;
#else
    return 0;
#endif
}

/* ------------------------------------------------------------------------
 * Metadata helpers
 * ------------------------------------------------------------------------ */

static void qwen4exp_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4: qwen4exp: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

/* Required u32 that accepts any of several spellings and refuses naming all
 * of them, so a metadata rename shows up as a readable message. */
static uint32_t qwen4exp_required_u32_any(const ds4_model *m,
                                          const char *const *keys,
                                          size_t nkeys) {
    uint32_t v = 0;
    if (model_get_u32_any(m, keys, nkeys, &v)) return v;
    fprintf(stderr, "ds4: qwen4exp: required metadata key is missing, tried:");
    for (size_t i = 0; i < nkeys; i++) fprintf(stderr, " %s", keys[i]);
    fprintf(stderr, "\n");
    exit(1);
}

/* Read a u32/i32/u64/i64 GGUF array into out[]; false when the key is absent. */
static bool qwen4exp_read_int_array(const ds4_model *m,
                                    const char *key,
                                    uint64_t *out,
                                    uint32_t cap,
                                    uint32_t *n_out) {
    ds4_array_ref arr = {0};
    if (!model_get_array(m, key, &arr)) return false;

    if (arr.len > cap) {
        qwen4exp_die("%s holds %" PRIu64 " entries, this build accepts %u",
                     key, arr.len, cap);
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        uint64_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            uint32_t v = 0;
            if (!cursor_u32(&c, &v)) ds4_die(c.error);
            got = v;
        } else if (arr.type == GGUF_VALUE_INT32) {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) qwen4exp_die("%s contains a negative value", key);
            got = (uint64_t)v;
        } else if (arr.type == GGUF_VALUE_UINT64) {
            if (!cursor_u64(&c, &got)) ds4_die(c.error);
        } else if (arr.type == GGUF_VALUE_INT64) {
            int64_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) qwen4exp_die("%s contains a negative value", key);
            got = (uint64_t)v;
        } else {
            qwen4exp_die("%s is not an integer array", key);
        }
        out[i] = got;
    }
    *n_out = (uint32_t)arr.len;
    return true;
}

static bool qwen4exp_read_int_array_any(const ds4_model *m,
                                        const char *const *keys,
                                        size_t nkeys,
                                        uint64_t *out,
                                        uint32_t cap,
                                        uint32_t *n_out) {
    for (size_t i = 0; i < nkeys; i++) {
        if (qwen4exp_read_int_array(m, keys[i], out, cap, n_out)) return true;
    }
    return false;
}

/* ------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------ */

/* Vocabulary size.  The unsloth artifact has no qwen4exp.vocab_size and only
 * the tokenizer array carries it; Baekpica has both.  When both are present
 * and disagree the file is inconsistent and we refuse rather than choose. */
static uint32_t qwen4exp_vocab_size(const ds4_model *m) {
    uint32_t kv_vocab = 0;
    const bool have_kv = model_get_u32(m, "qwen4exp.vocab_size", &kv_vocab);

    uint32_t tok_vocab = 0;
    ds4_array_ref arr = {0};
    const bool have_tok = model_get_array(m, "tokenizer.ggml.tokens", &arr);
    if (have_tok) {
        if (arr.len > UINT32_MAX) qwen4exp_die("tokenizer.ggml.tokens is too long");
        tok_vocab = (uint32_t)arr.len;
    }

    if (have_kv && have_tok && kv_vocab != tok_vocab) {
        qwen4exp_die("qwen4exp.vocab_size=%u disagrees with "
                     "tokenizer.ggml.tokens length %u",
                     kv_vocab, tok_vocab);
    }
    if (have_kv) return kv_vocab;
    if (have_tok) return tok_vocab;
    qwen4exp_die("neither qwen4exp.vocab_size nor tokenizer.ggml.tokens is present");
    return 0;
}

/* The QSA/GDN schedule as the file records it, cross-checked against the
 * full_attention_interval rule.  Refuses on the first block that disagrees. */
static void qwen4exp_validate_layer_schedule(const ds4_model *m, uint32_t n_layer) {
    uint64_t ratios[DS4_MAX_LAYER];
    uint32_t n = 0;
    static const char *const ratio_keys[] = {
        "qwen4exp.attention.compress_ratios",
    };
    static const char *const full_keys[] = {
        "qwen4exp.attention.full_layers",
    };

    if (qwen4exp_read_int_array_any(m, ratio_keys,
                                    sizeof(ratio_keys) / sizeof(ratio_keys[0]),
                                    ratios, DS4_MAX_LAYER, &n)) {
        if (n < n_layer) {
            qwen4exp_die("qwen4exp.attention.compress_ratios holds %u entries "
                         "but the model has %u blocks", n, n_layer);
        }
        for (uint32_t il = 0; il < n_layer; il++) {
            const bool qsa = ds4_qwen4exp_layer_is_full_attention(il);
            const uint64_t expected =
                qsa ? (uint64_t)DS4_N_INDEXER_COMPRESS_RATIO : 0ull;
            if (ratios[il] != expected) {
                qwen4exp_die("block %u has compress ratio %" PRIu64
                             ", expected %" PRIu64
                             " for a %s block at interval %u",
                             il, ratios[il], expected,
                             qsa ? "full-attention" : "linear-attention",
                             DS4_N_FULL_ATTN_INTERVAL);
            }
        }
        return;
    }

    if (qwen4exp_read_int_array_any(m, full_keys,
                                    sizeof(full_keys) / sizeof(full_keys[0]),
                                    ratios, DS4_MAX_LAYER, &n)) {
        uint32_t seen = 0;
        for (uint32_t il = 0; il < n_layer; il++) {
            if (ds4_qwen4exp_layer_is_full_attention(il)) seen++;
        }
        if (n != seen) {
            qwen4exp_die("qwen4exp.attention.full_layers lists %u blocks but the "
                         "interval-%u schedule has %u over %u blocks",
                         n, DS4_N_FULL_ATTN_INTERVAL, seen, n_layer);
        }
        for (uint32_t i = 0; i < n; i++) {
            if (ratios[i] >= n_layer ||
                !ds4_qwen4exp_layer_is_full_attention((uint32_t)ratios[i])) {
                qwen4exp_die("qwen4exp.attention.full_layers names block %" PRIu64
                             ", which is not a full-attention block at interval %u",
                             ratios[i], DS4_N_FULL_ATTN_INTERVAL);
            }
        }
        return;
    }

    qwen4exp_die("no attention schedule metadata: expected "
                 "qwen4exp.attention.compress_ratios or "
                 "qwen4exp.attention.full_layers");
}

/* Which block carries the ple_* tensors.
 *
 * The track fixture records target.ple_layer_ids ONE-BASED ([2]); the GGUF
 * records it zero-based ([1]).  The Baekpica artifact carries both spellings
 * and proves the convention, so when a file gives both we require they agree. */
static uint32_t qwen4exp_ple_layer(const ds4_model *m, uint32_t n_layer) {
    uint64_t zero_based[DS4_MAX_LAYER];
    uint64_t one_based[DS4_MAX_LAYER];
    uint32_t n_zero = 0, n_one = 0;

    static const char *const zero_keys[] = {
        "qwen4exp.ple.layers",
        "qwen4exp.ple.checkpoint_layer_ids_zero_based",
    };
    static const char *const one_keys[] = {
        "qwen4exp.ple.layer_ids_one_based",
    };

    if (!qwen4exp_read_int_array_any(m, zero_keys,
                                     sizeof(zero_keys) / sizeof(zero_keys[0]),
                                     zero_based, DS4_MAX_LAYER, &n_zero)) {
        qwen4exp_die("no PLE block metadata: expected qwen4exp.ple.layers or "
                     "qwen4exp.ple.checkpoint_layer_ids_zero_based");
    }
    if (n_zero != 1) {
        qwen4exp_die("this build supports exactly one PLE block, the file names %u",
                     n_zero);
    }
    if (zero_based[0] >= n_layer) {
        qwen4exp_die("PLE block %" PRIu64 " is outside the %u blocks of this model",
                     zero_based[0], n_layer);
    }

    if (qwen4exp_read_int_array_any(m, one_keys,
                                    sizeof(one_keys) / sizeof(one_keys[0]),
                                    one_based, DS4_MAX_LAYER, &n_one)) {
        if (n_one != n_zero || one_based[0] != zero_based[0] + 1ull) {
            qwen4exp_die("qwen4exp.ple.layer_ids_one_based=%" PRIu64
                         " does not match the zero-based PLE block %" PRIu64,
                         n_one ? one_based[0] : 0ull, zero_based[0]);
        }
    }
    return (uint32_t)zero_based[0];
}

/* Per-n-gram-head vocabulary partition.  rows must equal the sum of the head
 * vocabularies and every head must start where the previous one ended. */
static void qwen4exp_validate_ple_heads(const ds4_model *m,
                                        uint64_t rows,
                                        ds4_qwen4exp_ple_table *ple) {
    uint64_t offsets[DS4_QWEN4EXP_MAX_PLE_HEADS];
    uint64_t vocabs[DS4_QWEN4EXP_MAX_PLE_HEADS];
    uint32_t n_off = 0, n_voc = 0;

    const bool have_off = qwen4exp_read_int_array(
            m, "qwen4exp.ple.head_offsets", offsets,
            DS4_QWEN4EXP_MAX_PLE_HEADS, &n_off);
    const bool have_voc = qwen4exp_read_int_array(
            m, "qwen4exp.ple.head_vocab_sizes", vocabs,
            DS4_QWEN4EXP_MAX_PLE_HEADS, &n_voc);

    if (!have_off || !have_voc) {
        /* Artifacts that describe the partition by rule rather than by table
         * (ngram_vocab_size_base / ngram_parts) still have to agree on the
         * head count, which is the only thing the binder needs. */
        ple->heads = DS4_N_PLE_HEAD;
        return;
    }
    if (n_off != n_voc) {
        qwen4exp_die("qwen4exp.ple.head_offsets has %u entries and "
                     "qwen4exp.ple.head_vocab_sizes has %u", n_off, n_voc);
    }
    config_expect_u32("ple.head count", n_off, DS4_N_PLE_HEAD);

    uint64_t running = 0;
    for (uint32_t h = 0; h < n_off; h++) {
        if (offsets[h] != running) {
            qwen4exp_die("PLE head %u starts at row %" PRIu64
                         " but the previous heads end at row %" PRIu64,
                         h, offsets[h], running);
        }
        if (vocabs[h] == 0) qwen4exp_die("PLE head %u has an empty vocabulary", h);
        running += vocabs[h];
        ple->head_offset[h] = offsets[h];
        ple->head_vocab[h] = vocabs[h];
    }
    /*
     * The shipped table is taller than the head partition: the unsloth
     * artifact has 320,001,536 rows for 320,001,446 rows of vocabulary, i.e.
     * the height is rounded up to a 128/256 row boundary.  Accept a pad, never
     * a shortfall, and never more pad than one boundary's worth.
     */
    if (running > rows) {
        qwen4exp_die("the PLE head vocabularies sum to %" PRIu64
                     " rows but per_layer_token_embd.weight has only %" PRIu64,
                     running, rows);
    }
    if (rows - running >= DS4_QWEN4EXP_PLE_ROW_PAD) {
        qwen4exp_die("per_layer_token_embd.weight has %" PRIu64 " rows but the "
                     "head vocabularies only cover %" PRIu64
                     " (%" PRIu64 " unexplained rows, pad limit %u)",
                     rows, running, rows - running,
                     (unsigned)DS4_QWEN4EXP_PLE_ROW_PAD);
    }
    ple->covered_rows = running;
    ple->heads = n_off;
}

/*
 * The n-gram hash constants, read from the artifact and CHECKED against the
 * reference derivation.
 *
 * The engine keeps consuming the key-values: they are what the shipped table
 * was built with, and re-deriving them would make a wrongly converted file
 * silently produce a different set of rows.  The derivation is only the
 * tripwire.  It is the model implementation's own rule -- seed 1234, base
 * seed `seed + 10007 * ordinal`, a splitmix64 stream, `2 * (x % half) + 1` --
 * and it lives in ds4_qwen4exp_ple.c beside the hash it describes.
 *
 * The per-head vocabularies are checked by their rule rather than against a
 * base this artifact does not record: the reference takes consecutive primes,
 * so the sixteen sizes must be sixteen consecutive primes.
 *
 * A file with no multipliers is a reduced synthetic fixture and carries no
 * n-gram table worth hashing; it leaves `have_hash` false and the graph
 * refuses by name.
 */
static void qwen4exp_read_ple_hash_constants(const ds4_model *m,
                                             ds4_qwen4exp_ple_table *ple) {
    uint64_t multipliers[DS4_QWEN4EXP_MAX_NGRAM];
    uint32_t n_mult = 0;
    if (!qwen4exp_read_int_array(m, "qwen4exp.ple.layer_multipliers",
                                 multipliers, DS4_QWEN4EXP_MAX_NGRAM,
                                 &n_mult)) {
        return;
    }

    const uint32_t ngram = required_u32(m, "qwen4exp.ple.ngram_size");
    if (n_mult != ngram) {
        qwen4exp_die("QWEN4EXP-NGRAM-HASH-CONSTANTS-MISMATCH: "
                     "qwen4exp.ple.layer_multipliers holds %u entries and "
                     "qwen4exp.ple.ngram_size is %u", n_mult, ngram);
    }

    uint32_t eos = 0;
    if (!model_get_u32(m, "qwen4exp.ple.eos_token_id", &eos)) {
        qwen4exp_die("QWEN4EXP-NGRAM-HASH-CONSTANTS-MISMATCH: "
                     "qwen4exp.ple.layer_multipliers is present but "
                     "qwen4exp.ple.eos_token_id is missing, so the history "
                     "rule has no boundary token");
    }

    /* The ordinal is the block's POSITION in ple_layer_ids and not its layer
     * number: that position is what seeds the stream. */
    uint64_t layers[DS4_MAX_LAYER];
    uint32_t n_layers = 0;
    uint32_t ordinal = 0;
    static const char *const layer_keys[] = {
        "qwen4exp.ple.layers",
        "qwen4exp.ple.checkpoint_layer_ids_zero_based",
    };
    if (qwen4exp_read_int_array_any(m, layer_keys,
                                    sizeof(layer_keys) / sizeof(layer_keys[0]),
                                    layers, DS4_MAX_LAYER, &n_layers)) {
        for (uint32_t i = 0; i < n_layers; i++) {
            if (layers[i] == (uint64_t)g_ds4_shape.n_ple_layer) {
                ordinal = i;
                break;
            }
        }
    }

    int64_t seed = DS4_PLE_HASH_SEED_DEFAULT;
    uint32_t file_seed = 0;
    if (model_get_u32(m, "qwen4exp.ple.seed", &file_seed)) {
        seed = (int64_t)file_seed;
    }

    uint64_t derived[DS4_QWEN4EXP_MAX_NGRAM];
    ds4_ple_derive_multipliers(ngram, qwen4exp_vocab_size(m), ordinal, seed,
                               derived);
    for (uint32_t i = 0; i < ngram; i++) {
        if (derived[i] != multipliers[i]) {
            qwen4exp_die("QWEN4EXP-NGRAM-HASH-CONSTANTS-MISMATCH: "
                         "qwen4exp.ple.layer_multipliers[%u] is %" PRIu64
                         " and seed %lld at ple_layer_ids position %u over a "
                         "%u-token vocabulary derives %" PRIu64,
                         i, multipliers[i], (long long)seed, ordinal,
                         qwen4exp_vocab_size(m), derived[i]);
        }
    }

    if (ple->heads != 0 && ple->head_vocab[0] != 0 &&
        !ds4_ple_head_vocab_follows_rule(ple->head_vocab, ple->heads)) {
        qwen4exp_die("QWEN4EXP-NGRAM-HASH-CONSTANTS-MISMATCH: "
                     "the %u qwen4exp.ple.head_vocab_sizes are not %u "
                     "consecutive primes, which is the rule the reference "
                     "partitions the table by (first %" PRIu64 ")",
                     ple->heads, ple->heads, ple->head_vocab[0]);
    }

    ple->have_hash = true;
    ple->ngram_size = ngram;
    ple->eos_token_id = (int32_t)eos;
    for (uint32_t i = 0; i < ngram; i++) ple->multipliers[i] = multipliers[i];
}

static void qwen4exp_expect_shared_geometry(const ds4_model *m);
static void qwen4exp_finish_derived(uint32_t n_layer, uint32_t n_vocab,
                                    uint32_t ple_layer);

/*
 * Adopt the qwen4exp shape and refuse by name on any disagreement between the
 * GGUF metadata and the fixture geometry baked into DS4_SHAPE_QWEN4EXP.
 *
 * A file may declare qwen4exp.ds4.synthetic_reduced=true, which is how the
 * reduced-layer test artifact from tools/qwen4exp_synthetic_gguf.py says it is
 * not a production checkpoint.  Only the block count, the vocabulary and the
 * n-gram table height may shrink; every per-block width stays production
 * exact, which is the whole point of the test file.
 */
static void config_validate_qwen4exp_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_QWEN4EXP;
    memset(&g_ds4_qwen4exp, 0, sizeof(g_ds4_qwen4exp));

    bool synthetic = false;
    model_get_bool(m, "qwen4exp.ds4.synthetic_reduced", &synthetic);
    g_ds4_qwen4exp.synthetic_reduced = synthetic;

    const uint32_t n_layer = required_u32(m, "qwen4exp.block_count");
    const uint32_t n_vocab = qwen4exp_vocab_size(m);

    if (synthetic) {
        if (n_layer == 0 || n_layer > DS4_MAX_LAYER) {
            qwen4exp_die("reduced test model declares %u blocks, expected 1..%u",
                         n_layer, (uint32_t)DS4_MAX_LAYER);
        }
        if (n_vocab == 0) qwen4exp_die("reduced test model has an empty vocabulary");
        g_ds4_shape.n_layer = n_layer;
        g_ds4_shape.n_vocab = n_vocab;
        ds4_log(stderr, DS4_LOG_WARNING,
                "ds4: qwen4exp: this is a REDUCED test artifact "
                "(%u blocks, vocab %u), not a production checkpoint\n",
                n_layer, n_vocab);
    } else {
        config_expect_u32("block_count", n_layer, DS4_N_LAYER);
        config_expect_u32("vocab_size", n_vocab, DS4_N_VOCAB);
    }

    qwen4exp_expect_shared_geometry(m);

    qwen4exp_validate_layer_schedule(m, n_layer);

    const uint32_t ple_layer = qwen4exp_ple_layer(m, n_layer);
    if (!synthetic) {
        config_expect_u32("ple block", ple_layer, DS4_N_PLE_LAYER);
    }
    g_ds4_shape.n_ple_layer = ple_layer;

    /* Per-layer n-gram embedding: target model only, the MTP head has none. */
    config_expect_u32("ple.ngram_size",
                      required_u32(m, "qwen4exp.ple.ngram_size"), DS4_N_NGRAM);
    config_expect_u32("ple.heads_per_ngram",
                      required_u32(m, "qwen4exp.ple.heads_per_ngram"),
                      DS4_N_NGRAM_HEADS_PER);
    config_expect_u32("ple.conv_kernel",
                      required_u32(m, "qwen4exp.ple.conv_kernel"), DS4_N_PLE_CONV);
    {
        static const char *const keys[] = {
            "qwen4exp.embedding_length_per_layer_input",
            "qwen4exp.ple.input_length",
        };
        config_expect_u32("embedding_length_per_layer_input",
                          qwen4exp_required_u32_any(m, keys, 2),
                          DS4_N_PLE_ROW_DIM);
    }
    {
        uint32_t ple_embd = 0;
        if (model_get_u32(m, "qwen4exp.ple.embedding_length", &ple_embd)) {
            config_expect_u32("ple.embedding_length", ple_embd, DS4_N_PLE_EMBD);
        }
    }

    qwen4exp_finish_derived(n_layer, n_vocab, ple_layer);
}

/*
 * The geometry both the target model and the MTP head must agree on.  Split
 * out so the head, which has no PLE, no tokenizer and one extra block, is
 * checked against exactly the same numbers.
 */
static void qwen4exp_expect_shared_geometry(const ds4_model *m) {
    config_expect_u32("embedding_length",
                      required_u32(m, "qwen4exp.embedding_length"), DS4_N_EMBD);
    config_expect_u32("attention.head_count",
                      required_u32(m, "qwen4exp.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv",
                      required_u32(m, "qwen4exp.attention.head_count_kv"),
                      DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length",
                      required_u32(m, "qwen4exp.attention.key_length"),
                      DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length",
                      required_u32(m, "qwen4exp.attention.value_length"),
                      DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",
                      required_u32(m, "qwen4exp.rope.dimension_count"), DS4_N_ROT);
    config_expect_u32("expert_count",
                      required_u32(m, "qwen4exp.expert_count"), DS4_N_EXPERT);
    config_expect_u32("expert_used_count",
                      required_u32(m, "qwen4exp.expert_used_count"),
                      DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length",
                      required_u32(m, "qwen4exp.expert_feed_forward_length"),
                      DS4_N_FF_EXP);
    config_expect_u32("expert_shared_feed_forward_length",
                      required_u32(m, "qwen4exp.expert_shared_feed_forward_length"),
                      DS4_N_FF_SHEXP);
    config_expect_u32("full_attention_interval",
                      required_u32(m, "qwen4exp.full_attention_interval"),
                      DS4_N_FULL_ATTN_INTERVAL);
    config_expect_u32("attention.indexer.head_count",
                      required_u32(m, "qwen4exp.attention.indexer.head_count"),
                      DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length",
                      required_u32(m, "qwen4exp.attention.indexer.key_length"),
                      DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",
                      required_u32(m, "qwen4exp.attention.indexer.top_k"),
                      DS4_N_INDEXER_TOP_K);
    config_expect_f32("rope.freq_base",
                      required_f32(m, "qwen4exp.rope.freq_base"),
                      DS4_ROPE_FREQ_BASE);
    config_expect_epsilon("attention.layer_norm_rms_epsilon",
                          required_f32(m, "qwen4exp.attention.layer_norm_rms_epsilon"),
                          DS4_RMS_EPS);

    /* Hyper connections: two spellings of the low-rank width. */
    config_expect_u32("hyper_connection.count",
                      required_u32(m, "qwen4exp.hyper_connection.count"), DS4_N_HC);
    {
        static const char *const keys[] = {
            "qwen4exp.hyper_connection.low_rank",
            "qwen4exp.hyper_connection.lowrank",
        };
        config_expect_u32("hyper_connection.low_rank",
                          qwen4exp_required_u32_any(m, keys,
                                                    sizeof(keys) / sizeof(keys[0])),
                          DS4_N_HC_LOWRANK);
    }

    /* Gated delta net: the unsloth file uses the ssm.* names, Baekpica the
     * attention.linear_* names.  Both must land on the same widths. */
    {
        static const char *const key_head[] = {
            "qwen4exp.ssm.group_count",
            "qwen4exp.attention.linear_key_head_count",
        };
        static const char *const value_head[] = {
            "qwen4exp.ssm.time_step_rank",
            "qwen4exp.attention.linear_value_head_count",
        };
        static const char *const state[] = {
            "qwen4exp.ssm.state_size",
            "qwen4exp.attention.linear_key_length",
        };
        static const char *const conv[] = {
            "qwen4exp.ssm.conv_kernel",
            "qwen4exp.attention.linear_conv_kernel",
        };
        config_expect_u32("ssm.group_count",
                          qwen4exp_required_u32_any(m, key_head, 2),
                          DS4_N_GDN_KEY_HEAD);
        config_expect_u32("ssm.time_step_rank",
                          qwen4exp_required_u32_any(m, value_head, 2),
                          DS4_N_GDN_VALUE_HEAD);
        config_expect_u32("ssm.state_size",
                          qwen4exp_required_u32_any(m, state, 2),
                          DS4_N_GDN_STATE);
        config_expect_u32("ssm.conv_kernel",
                          qwen4exp_required_u32_any(m, conv, 2),
                          DS4_N_GDN_CONV);

        uint32_t inner = 0;
        if (model_get_u32(m, "qwen4exp.ssm.inner_size", &inner)) {
            config_expect_u32("ssm.inner_size", inner, DS4_N_GDN_INNER);
        }
        uint32_t value_len = 0;
        if (model_get_u32(m, "qwen4exp.attention.linear_value_length", &value_len)) {
            config_expect_u32("attention.linear_value_length", value_len,
                              DS4_N_GDN_HEAD_DIM);
        }
    }

}

/* Widths L3-L7 read instead of recomputing. */
static void qwen4exp_finish_derived(uint32_t n_layer, uint32_t n_vocab,
                                    uint32_t ple_layer) {
#ifndef DS4_NO_GPU
    /* This tower's dense Q8_0 projections take the int8 tensor-core GEMM.  The
     * entry they go through is shared with the GLM and DeepSeek towers, whose
     * numbers are pinned against the eight-row tile's reduction order, so the
     * switch is per family and this is where the family is known. */
    ds4_gpu_enable_q8_dense_mma();
#endif
    g_ds4_qwen4exp.n_layer = n_layer;
    g_ds4_qwen4exp.n_vocab = n_vocab;
    g_ds4_qwen4exp.ple_layer = ple_layer;
    g_ds4_qwen4exp.hc_dim = DS4_N_HC * DS4_N_EMBD;
    g_ds4_qwen4exp.gdn_inner = DS4_N_GDN_VALUE_HEAD * DS4_N_GDN_HEAD_DIM;
    g_ds4_qwen4exp.gdn_qkv_dim =
        (2u * DS4_N_GDN_KEY_HEAD + DS4_N_GDN_VALUE_HEAD) * DS4_N_GDN_HEAD_DIM;
    g_ds4_qwen4exp.qsa_q_dim = DS4_N_HEAD * DS4_N_HEAD_DIM * 2u;
    g_ds4_qwen4exp.qsa_kv_dim = DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    g_ds4_qwen4exp.indexer_q_dim = DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    g_ds4_qwen4exp.indexer_k_dim =
        DS4_N_INDEXER_KV_HEAD * DS4_N_INDEXER_HEAD_DIM;

    config_expect_u32("ssm.inner_size (derived)", g_ds4_qwen4exp.gdn_inner,
                      DS4_N_GDN_INNER);
}

/*
 * The MTP head is a second GGUF with one extra block and no PLE, no tokenizer
 * and no embedding: it borrows the target's.  It must agree with the target on
 * every shared width, so it runs the same geometry check.
 */
/* Reachable from the DS4_TEST_HOOKS entry points below and from the engine
 * open path once L5 wires it; upstream marks the same shape of function with
 * DS4_MAYBE_UNUSED (see weights_model_map_sharded_spans in ds4.c). */
static DS4_MAYBE_UNUSED void config_validate_qwen4exp_mtp_model(const ds4_model *m) {
    const uint32_t n_predict = required_u32(m, "qwen4exp.nextn_predict_layers");
    if (n_predict != 1) {
        qwen4exp_die("the MTP head declares nextn_predict_layers=%u, "
                     "this build supports exactly 1", n_predict);
    }
    if (!required_bool(m, "qwen4exp.nextn_shared_target_tensors")) {
        qwen4exp_die("the MTP head does not set nextn_shared_target_tensors; "
                     "this build needs a head that borrows the target's "
                     "token_embd and output");
    }
    config_expect_u32("MTP block_count",
                      required_u32(m, "qwen4exp.block_count"),
                      DS4_N_LAYER + n_predict);

    qwen4exp_expect_shared_geometry(m);
    qwen4exp_finish_derived(DS4_N_LAYER, g_ds4_shape.n_vocab,
                            g_ds4_shape.n_ple_layer);
}

/* ------------------------------------------------------------------------
 * Tensor binding
 * ------------------------------------------------------------------------ */

#define DS4_QWEN4EXP_MAX_TYPES 6

typedef struct {
    const char *name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type[DS4_QWEN4EXP_MAX_TYPES];
    uint32_t n_type;
    ds4_qwen4exp_mem_family family;
} qwen4exp_tensor_spec;

static void qwen4exp_plan_charge(ds4_qwen4exp_mem_plan *plan,
                                 ds4_qwen4exp_mem_family family,
                                 const ds4_tensor *t) {
    plan->bytes[family] += t->bytes;
    plan->tensors[family] += 1;
    plan->bound_tensors += 1;
}

/*
 * Find one tensor and refuse by name unless its type, rank and every dimension
 * are what this family declares.  The accepted type list is a set because the
 * shipped mixed-quant artifacts vary the routed-expert types per block.
 */
static ds4_tensor *qwen4exp_bind(const ds4_model *m,
                                 ds4_qwen4exp_mem_plan *plan,
                                 qwen4exp_tensor_spec spec_v) {
    const qwen4exp_tensor_spec *spec = &spec_v;
    ds4_tensor *t = model_find_tensor(m, spec->name);
    if (!t) {
        qwen4exp_die("required tensor is missing: %s", spec->name);
    }

    bool type_ok = false;
    for (uint32_t i = 0; i < spec->n_type; i++) {
        if (t->type == spec->type[i]) { type_ok = true; break; }
    }
    if (!type_ok) {
        char accepted[256];
        size_t used = 0;
        accepted[0] = '\0';
        for (uint32_t i = 0; i < spec->n_type && used < sizeof(accepted) - 1; i++) {
            const int n = snprintf(accepted + used, sizeof(accepted) - used,
                                   "%s%s", i ? "/" : "",
                                   tensor_type_name(spec->type[i]));
            if (n < 0) break;
            used += (size_t)n;
        }
        qwen4exp_die("tensor %s has type %s, expected %s",
                     spec->name, tensor_type_name(t->type), accepted);
    }

    if (t->ndim != spec->ndim) {
        qwen4exp_die("tensor %s has rank %u, expected %u",
                     spec->name, t->ndim, spec->ndim);
    }
    for (uint32_t d = 0; d < spec->ndim; d++) {
        if (t->dim[d] != spec->dim[d]) {
            qwen4exp_die("tensor %s has dim[%u]=%" PRIu64 ", expected %" PRIu64,
                         spec->name, d, t->dim[d], spec->dim[d]);
        }
    }

    qwen4exp_plan_charge(plan, spec->family, t);
    return t;
}

/* Convenience wrappers so the per-layer table below reads as a shape list. */
#define QW_SPEC1(nm, t0, d0, fam) \
    ((qwen4exp_tensor_spec){ .name = (nm), .ndim = 1, .dim = { (d0) }, \
                             .type = { (t0) }, .n_type = 1, .family = (fam) })
#define QW_SPEC2(nm, t0, d0, d1, fam) \
    ((qwen4exp_tensor_spec){ .name = (nm), .ndim = 2, .dim = { (d0), (d1) }, \
                             .type = { (t0) }, .n_type = 1, .family = (fam) })
#define QW_SPEC3(nm, t0, d0, d1, d2, fam) \
    ((qwen4exp_tensor_spec){ .name = (nm), .ndim = 3, \
                             .dim = { (d0), (d1), (d2) }, \
                             .type = { (t0) }, .n_type = 1, .family = (fam) })

static void qwen4exp_name(char *buf, size_t size, const char *fmt, uint32_t il) {
    const int n = snprintf(buf, size, fmt, il);
    if (n < 0 || (size_t)n >= size) ds4_die("tensor name is too long");
}

/* Expert weights ship as a per-block mix, so a role takes a set of types
 * rather than one.  The set is ds4_qwen4exp_moe_types.h, which the Metal and
 * CUDA expert GEMMs expand into their dequant switch: the loader therefore
 * accepts exactly what the kernels decode, and refuses anything else by name.
 */
#define QW_EXP_TYPE_ID(name, id) (uint32_t)(id),
static const uint32_t QW_EXP_TYPES[] = {
    DS4_QWEN4EXP_MOE_TYPES(QW_EXP_TYPE_ID)
};
#undef QW_EXP_TYPE_ID
static const uint32_t QW_EMBED_TYPES[] = { DS4_TENSOR_Q8_0, DS4_TENSOR_Q6_K };
static const uint32_t QW_INDEXER_TYPES[] = { DS4_TENSOR_BF16, DS4_TENSOR_F16 };
/* From the same table the inject kernels switch on, so the accepted set and the
 * decodable set are one thing.  See ds4_qwen4exp_hc_types.h. */
#define QW_HC_INJECT_TYPE_ID(name, id) (uint32_t)(id),
static const uint32_t QW_HC_INJECT_TYPES[] = {
    DS4_QWEN4EXP_HC_INJECT_TYPES(QW_HC_INJECT_TYPE_ID)
};
#undef QW_HC_INJECT_TYPE_ID
static const uint32_t QW_PLE_TABLE_TYPES[] = {
    DS4_TENSOR_IQ4_NL, DS4_TENSOR_IQ4_XS, DS4_TENSOR_BF16,
    DS4_TENSOR_F16, DS4_TENSOR_Q8_0,
};

static qwen4exp_tensor_spec qwen4exp_spec_types(qwen4exp_tensor_spec spec,
                                                const uint32_t *types,
                                                uint32_t n_types) {
    if (n_types > DS4_QWEN4EXP_MAX_TYPES) ds4_die("too many accepted tensor types");
    for (uint32_t i = 0; i < n_types; i++) spec.type[i] = types[i];
    spec.n_type = n_types;
    return spec;
}

#define QW_TYPES(spec, arr) \
    qwen4exp_spec_types((spec), (arr), (uint32_t)(sizeof(arr) / sizeof((arr)[0])))

static void qwen4exp_bind_block_tensors(ds4_qwen4exp_layer_weights *l,
                                        const ds4_model *m,
                                        ds4_qwen4exp_mem_plan *plan,
                                        uint32_t il,
                                        bool full_attention,
                                        bool with_ple) {
    const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;
    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t hc_dim = cfg->hc_dim;
    char nm[128];

    memset(l, 0, sizeof(*l));
    l->is_full_attention = full_attention;
    l->has_ple = with_ple;

    if (!l->is_full_attention) {
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_qkv.weight", il);
        l->attn_qkv = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, cfg->gdn_qkv_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_gate.weight", il);
        l->attn_gate = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, cfg->gdn_inner, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_out.weight", il);
        l->ssm_out = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                cfg->gdn_inner, n_embd, DS4_QWEN4EXP_MEM_DENSE));

        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_conv1d.weight", il);
        l->ssm_conv1d = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_F32,
                DS4_N_GDN_CONV, cfg->gdn_qkv_dim, DS4_QWEN4EXP_MEM_GDN));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_alpha.weight", il);
        l->ssm_alpha = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_F32,
                n_embd, DS4_N_GDN_VALUE_HEAD, DS4_QWEN4EXP_MEM_GDN));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_beta.weight", il);
        l->ssm_beta = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_F32,
                n_embd, DS4_N_GDN_VALUE_HEAD, DS4_QWEN4EXP_MEM_GDN));
        /* ssm_a is -exp(A_log), NOT the raw A_log this family's GLM 5.3 KDA
         * sibling stores: llama.cpp's converter exponentiates and negates at
         * conversion time, so every value here is negative.  The kernels use it
         * as-is -- decay = exp(softplus(alpha + dt_bias) * ssm_a). */
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_a", il);
        l->ssm_a = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_GDN_VALUE_HEAD, DS4_QWEN4EXP_MEM_GDN));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_dt.bias", il);
        l->ssm_dt_bias = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_GDN_VALUE_HEAD, DS4_QWEN4EXP_MEM_GDN));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ssm_norm.weight", il);
        l->ssm_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_GDN_HEAD_DIM, DS4_QWEN4EXP_MEM_GDN));
    } else {
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_q.weight", il);
        l->attn_q = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, cfg->qsa_q_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_k.weight", il);
        l->attn_k = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, cfg->qsa_kv_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_v.weight", il);
        l->attn_v = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, cfg->qsa_kv_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_output.weight", il);
        l->attn_output = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM, n_embd,
                DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_q_norm.weight", il);
        l->attn_q_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_HEAD_DIM, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.attn_k_norm.weight", il);
        l->attn_k_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_HEAD_DIM, DS4_QWEN4EXP_MEM_DENSE));

        qwen4exp_name(nm, sizeof(nm), "blk.%u.indexer.q_proj.weight", il);
        l->indexer_q_proj = qwen4exp_bind(m, plan,
                QW_TYPES(QW_SPEC2(nm, 0, n_embd, cfg->indexer_q_dim,
                                   DS4_QWEN4EXP_MEM_INDEXER), QW_INDEXER_TYPES));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.indexer.k_proj.weight", il);
        l->indexer_k_proj = qwen4exp_bind(m, plan,
                QW_TYPES(QW_SPEC2(nm, 0, n_embd, cfg->indexer_k_dim,
                                   DS4_QWEN4EXP_MEM_INDEXER), QW_INDEXER_TYPES));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.indexer.q_norm.weight", il);
        l->indexer_q_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_INDEXER_HEAD_DIM, DS4_QWEN4EXP_MEM_INDEXER));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.indexer.k_norm.weight", il);
        l->indexer_k_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                DS4_N_INDEXER_HEAD_DIM, DS4_QWEN4EXP_MEM_INDEXER));
    }

    /* Mixture of experts and the shared expert: every block. */
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_gate_inp = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_F32,
            n_embd, DS4_N_EXPERT, DS4_QWEN4EXP_MEM_ROUTER));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_gate_exps = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC3(nm, 0, n_embd, DS4_N_FF_EXP, DS4_N_EXPERT,
                               DS4_QWEN4EXP_MEM_EXPERTS), QW_EXP_TYPES));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_up_exps.weight", il);
    l->ffn_up_exps = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC3(nm, 0, n_embd, DS4_N_FF_EXP, DS4_N_EXPERT,
                               DS4_QWEN4EXP_MEM_EXPERTS), QW_EXP_TYPES));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_down_exps.weight", il);
    l->ffn_down_exps = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC3(nm, 0, DS4_N_FF_EXP, n_embd, DS4_N_EXPERT,
                               DS4_QWEN4EXP_MEM_EXPERTS), QW_EXP_TYPES));

    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_gate_inp_shexp.weight", il);
    l->ffn_gate_inp_shexp = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            n_embd, DS4_QWEN4EXP_MEM_ROUTER));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_gate_shexp = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            n_embd, DS4_N_FF_SHEXP, DS4_QWEN4EXP_MEM_DENSE));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_up_shexp = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            n_embd, DS4_N_FF_SHEXP, DS4_QWEN4EXP_MEM_DENSE));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.ffn_down_shexp.weight", il);
    l->ffn_down_shexp = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            DS4_N_FF_SHEXP, n_embd, DS4_QWEN4EXP_MEM_DENSE));

    /* Hyper connections: every block, attention side and FFN side. */
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_attn_down.weight", il);
    l->hc_attn_down = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            hc_dim, DS4_N_HC_LOWRANK, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_attn_up.weight", il);
    l->hc_attn_up = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            DS4_N_HC_LOWRANK, hc_dim, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_attn_inject.weight", il);
    l->hc_attn_inject = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC2(nm, 0, hc_dim, DS4_N_HC,
                              DS4_QWEN4EXP_MEM_HC), QW_HC_INJECT_TYPES));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_attn_norm.weight", il);
    l->hc_attn_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            hc_dim, DS4_QWEN4EXP_MEM_HC));

    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_ffn_down.weight", il);
    l->hc_ffn_down = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            hc_dim, DS4_N_HC_LOWRANK, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_ffn_up.weight", il);
    l->hc_ffn_up = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            DS4_N_HC_LOWRANK, hc_dim, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_ffn_inject.weight", il);
    l->hc_ffn_inject = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC2(nm, 0, hc_dim, DS4_N_HC,
                              DS4_QWEN4EXP_MEM_HC), QW_HC_INJECT_TYPES));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.hc_ffn_norm.weight", il);
    l->hc_ffn_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            hc_dim, DS4_QWEN4EXP_MEM_HC));

    /* Per-layer n-gram embedding projections: only the PLE block. */
    if (l->has_ple) {
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_conv1d.weight", il);
        l->ple_conv1d = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_F32,
                DS4_N_PLE_CONV, hc_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_key.weight", il);
        l->ple_key = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, hc_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_value.weight", il);
        l->ple_value = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
                n_embd, DS4_N_PLE_EMBD, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_norm_conv.weight", il);
        l->ple_norm_conv = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                hc_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_norm_key.weight", il);
        l->ple_norm_key = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                hc_dim, DS4_QWEN4EXP_MEM_DENSE));
        qwen4exp_name(nm, sizeof(nm), "blk.%u.ple_norm_query.weight", il);
        l->ple_norm_query = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
                hc_dim, DS4_QWEN4EXP_MEM_DENSE));
    }
}

/* Bind one block of the target model, deriving its type from the schedule. */

/* ------------------------------------------------------------------------
 * Norm weight convention
 * ------------------------------------------------------------------------
 *
 * The MLX checkpoint stores `1 + w` for every RMS norm.  Whether the GGUF
 * converter kept that or wrote zero-centered `w` is not recorded anywhere in
 * the metadata, and both files validate: same name, same shape, same type,
 * same byte count.  Getting it wrong scales every normalised activation by
 * roughly 1 instead of roughly 2, or the reverse, and nothing refuses.
 *
 * Classify from the values, by the NEGATIVE FRACTION and not the mean (design
 * section 2).  A zero-centered `w` scatters about 0, so about half its entries
 * are negative; a baked `1 + w` clusters about 1 and few are.  The boundary is
 * 0.20.  The mean does not separate them: a baked tensor whose weights happen
 * to be small and a zero-centered one with a positive bias can land on the same
 * mean, while the sign census cannot.
 *
 * Returns the offset the kernels take: 1.0 for zero-centered, 0.0 for baked.
 * A tensor landing inside the ambiguous band around the boundary is refused by
 * name -- guessing there is exactly the silent failure this check exists for.
 */
#define DS4_QWEN4EXP_NORM_NEG_BOUNDARY   0.20f
#define DS4_QWEN4EXP_NORM_NEG_AMBIGUOUS  0.05f

static float qwen4exp_norm_weight_offset(const ds4_model *m,
                                         const ds4_tensor *t,
                                         const char *what) {
    if (!t) qwen4exp_die("%s: no norm tensor to classify", what);
    if (t->type != DS4_TENSOR_F32) {
        qwen4exp_die("%s: norm weights must be F32 to classify, got %s",
                     what, tensor_type_name(t->type));
    }
    if (t->elements == 0) qwen4exp_die("%s: norm tensor is empty", what);

    const float *v = (const float *)tensor_data(m, t);
    uint64_t negative = 0;
    for (uint64_t i = 0; i < t->elements; i++) {
        if (v[i] < 0.0f) negative++;
    }
    const float fraction = (float)negative / (float)t->elements;

    const float lo = DS4_QWEN4EXP_NORM_NEG_BOUNDARY - DS4_QWEN4EXP_NORM_NEG_AMBIGUOUS;
    const float hi = DS4_QWEN4EXP_NORM_NEG_BOUNDARY + DS4_QWEN4EXP_NORM_NEG_AMBIGUOUS;
    if (fraction > lo && fraction < hi) {
        qwen4exp_die("%s: %.1f%% of the norm weights are negative, inside the "
                     "%.0f-%.0f%% band where zero-centered and offset-baked "
                     "cannot be told apart",
                     what, (double)fraction * 100.0,
                     (double)lo * 100.0, (double)hi * 100.0);
    }
    return fraction > DS4_QWEN4EXP_NORM_NEG_BOUNDARY ? 1.0f : 0.0f;
}

static void weights_bind_qwen4exp_layer(ds4_qwen4exp_layer_weights *l,
                                        const ds4_model *m,
                                        ds4_qwen4exp_mem_plan *plan,
                                        uint32_t il) {
    qwen4exp_bind_block_tensors(l, m, plan, il,
                                ds4_qwen4exp_layer_is_full_attention(il),
                                il == g_ds4_qwen4exp.ple_layer);

    char what[96];
    snprintf(what, sizeof(what), "blk.%u.hc_attn_norm.weight", il);
    l->hc_attn_norm_offset = qwen4exp_norm_weight_offset(m, l->hc_attn_norm, what);
    snprintf(what, sizeof(what), "blk.%u.hc_ffn_norm.weight", il);
    l->hc_ffn_norm_offset = qwen4exp_norm_weight_offset(m, l->hc_ffn_norm, what);

    if (l->is_full_attention) {
        snprintf(what, sizeof(what), "blk.%u.attn_q_norm.weight", il);
        l->attn_q_norm_offset = qwen4exp_norm_weight_offset(m, l->attn_q_norm, what);
        snprintf(what, sizeof(what), "blk.%u.attn_k_norm.weight", il);
        l->attn_k_norm_offset = qwen4exp_norm_weight_offset(m, l->attn_k_norm, what);
        snprintf(what, sizeof(what), "blk.%u.indexer.q_norm.weight", il);
        l->indexer_q_norm_offset =
            qwen4exp_norm_weight_offset(m, l->indexer_q_norm, what);
        snprintf(what, sizeof(what), "blk.%u.indexer.k_norm.weight", il);
        l->indexer_k_norm_offset =
            qwen4exp_norm_weight_offset(m, l->indexer_k_norm, what);
    }

    if (l->has_ple) {
        snprintf(what, sizeof(what), "blk.%u.ple_norm_key.weight", il);
        l->ple_norm_key_offset =
            qwen4exp_norm_weight_offset(m, l->ple_norm_key, what);
        snprintf(what, sizeof(what), "blk.%u.ple_norm_query.weight", il);
        l->ple_norm_query_offset =
            qwen4exp_norm_weight_offset(m, l->ple_norm_query, what);
        snprintf(what, sizeof(what), "blk.%u.ple_norm_conv.weight", il);
        l->ple_norm_conv_offset =
            qwen4exp_norm_weight_offset(m, l->ple_norm_conv, what);
    }
}

/*
 * The n-gram table is bound as a handle over the mapping and never read.
 * The row stride comes from the tensor's own quantization, so an IQ4_NL table
 * gives 90 bytes per 160-value row and a BF16 sidecar gives 320.
 */
static void weights_bind_qwen4exp_ple_table(ds4_qwen4exp_ple_table *ple,
                                    const ds4_model *m,
                                    ds4_qwen4exp_mem_plan *plan) {
    memset(ple, 0, sizeof(*ple));

    ds4_tensor *t = model_find_tensor(m, "per_layer_token_embd.weight");
    if (!t) qwen4exp_die("required tensor is missing: per_layer_token_embd.weight");

    bool type_ok = false;
    for (size_t i = 0;
         i < sizeof(QW_PLE_TABLE_TYPES) / sizeof(QW_PLE_TABLE_TYPES[0]); i++) {
        if (t->type == QW_PLE_TABLE_TYPES[i]) { type_ok = true; break; }
    }
    if (!type_ok) {
        qwen4exp_die("tensor per_layer_token_embd.weight has type %s, expected "
                     "iq4_nl/iq4_xs/bf16/f16/q8_0", tensor_type_name(t->type));
    }
    if (t->ndim != 2) {
        qwen4exp_die("tensor per_layer_token_embd.weight has rank %u, expected 2",
                     t->ndim);
    }
    if (t->dim[0] != DS4_N_PLE_ROW_DIM) {
        qwen4exp_die("tensor per_layer_token_embd.weight has dim[0]=%" PRIu64
                     ", expected %u", t->dim[0], DS4_N_PLE_ROW_DIM);
    }

    uint64_t row_bytes = 0;
    if (!tensor_nbytes(t->type, t->dim[0], &row_bytes) || row_bytes == 0) {
        qwen4exp_die("cannot size a %s row of %" PRIu64 " values",
                     tensor_type_name(t->type), t->dim[0]);
    }
    if (row_bytes * t->dim[1] != t->bytes) {
        qwen4exp_die("per_layer_token_embd.weight is %" PRIu64 " bytes but "
                     "%" PRIu64 " rows of %" PRIu64 " bytes is %" PRIu64,
                     t->bytes, t->dim[1], row_bytes, row_bytes * t->dim[1]);
    }

    ple->tensor = t;
    ple->base = (const uint8_t *)tensor_data(m, t);
    ple->rows = t->dim[1];
    ple->row_dim = t->dim[0];
    ple->row_bytes = row_bytes;
    ple->bytes = t->bytes;
    ple->type = t->type;

    qwen4exp_validate_ple_heads(m, ple->rows, ple);
    qwen4exp_read_ple_hash_constants(m, ple);
    qwen4exp_plan_charge(plan, DS4_QWEN4EXP_MEM_PLE, t);
    g_ds4_qwen4exp.ple_rows = ple->rows;
}

static const char *qwen4exp_family_name(ds4_qwen4exp_mem_family f) {
    switch (f) {
    case DS4_QWEN4EXP_MEM_EXPERTS:  return "experts";
    case DS4_QWEN4EXP_MEM_PLE:      return "PLE (SSD)";
    case DS4_QWEN4EXP_MEM_DENSE:    return "dense";
    case DS4_QWEN4EXP_MEM_EMBED:    return "embed/output";
    case DS4_QWEN4EXP_MEM_HC:       return "HC";
    case DS4_QWEN4EXP_MEM_GDN:      return "GDN";
    case DS4_QWEN4EXP_MEM_ROUTER:   return "router";
    case DS4_QWEN4EXP_MEM_INDEXER:  return "indexer";
    default:                        return "unknown";
    }
}

static void qwen4exp_memory_plan_print(const ds4_model *m,
                                       const ds4_qwen4exp_mem_plan *plan,
                                       const ds4_qwen4exp_ple_table *ple) {
    printf("qwen4exp memory plan (%" PRIu64 " tensors over %u shard%s):\n",
           plan->bound_tensors, m->n_shards, m->n_shards == 1 ? "" : "s");
    for (int f = 0; f < DS4_QWEN4EXP_MEM_COUNT; f++) {
        printf("  %-13s %6" PRIu64 " tensors, ",
               qwen4exp_family_name((ds4_qwen4exp_mem_family)f),
               plan->tensors[f]);
        print_size(plan->bytes[f]);
        printf("\n");
    }
    printf("  %-13s %6s  ", "resident", "");
    print_size(plan->resident_bytes);
    printf("\n");
    printf("  %-13s %6s  ", "SSD resident", "");
    print_size(plan->ssd_bytes);
    printf(" (%" PRIu64 " rows x %" PRIu64 " %s, %" PRIu64 " B/row)\n",
           ple->rows, ple->row_dim, tensor_type_name(ple->type), ple->row_bytes);
    printf("  %-13s %6s  ", "total", "");
    print_size(plan->total_bytes);
    printf("\n");
}

/*
 * MEMORY SAFETY.  Nothing above allocated model memory: every slot is a
 * pointer into the mapping.  Before any caller starts making the resident set
 * real, refuse when free unified memory cannot hold it plus a fixed headroom.
 */
static void qwen4exp_memory_guard(const ds4_qwen4exp_mem_plan *plan) {
    const uint64_t headroom = DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES;
    const uint64_t free_bytes = qwen4exp_free_memory_bytes();

    if (free_bytes == 0) {
        ds4_log(stderr, DS4_LOG_WARNING,
                "ds4: qwen4exp: cannot read free unified memory on this "
                "platform; the load is not budget checked\n");
        return;
    }
    if (plan->resident_bytes > UINT64_MAX - headroom) {
        qwen4exp_die("resident byte total overflows the memory budget");
    }

    const uint64_t required = plan->resident_bytes + headroom;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    if (free_bytes < required) {
        fprintf(stderr,
                "ds4: qwen4exp: refusing to load: free unified memory is "
                "%.2f GiB but this model needs %.2f GiB resident plus "
                "%.2f GiB headroom (%.2f GiB). Free memory, or run a "
                "smaller quantization.\n",
                (double)free_bytes / gib,
                (double)plan->resident_bytes / gib,
                (double)headroom / gib,
                (double)required / gib);
        exit(1);
    }

    printf("qwen4exp memory budget: free ");
    print_size(free_bytes);
    printf(", resident ");
    print_size(plan->resident_bytes);
    printf(", headroom ");
    print_size(headroom);
    printf(" -- ok\n");
}

/*
 * Bind every tensor of a qwen4exp checkpoint, print the plan and run the
 * memory guard.  Refuses by name on the first tensor that is missing, has the
 * wrong quantization, or has the wrong shape.
 */
/* Reachable from the DS4_TEST_HOOKS entry points below and from the engine
 * open path once L5 wires it; upstream marks the same shape of function with
 * DS4_MAYBE_UNUSED (see weights_model_map_sharded_spans in ds4.c). */
static DS4_MAYBE_UNUSED void weights_bind_qwen4exp(ds4_qwen4exp_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->n_layer = g_ds4_qwen4exp.n_layer;

    ds4_qwen4exp_mem_plan *plan = &w->plan;

    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t n_vocab = g_ds4_qwen4exp.n_vocab;
    const uint64_t hc_dim = g_ds4_qwen4exp.hc_dim;

    w->token_embd = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC2("token_embd.weight", 0, n_embd, n_vocab,
                               DS4_QWEN4EXP_MEM_EMBED), QW_EMBED_TYPES));
    w->output = qwen4exp_bind(m, plan,
            QW_TYPES(QW_SPEC2("output.weight", 0, n_embd, n_vocab,
                               DS4_QWEN4EXP_MEM_EMBED), QW_EMBED_TYPES));
    w->output_hc_down = qwen4exp_bind(m, plan,
            QW_SPEC2("output_hc_down.weight", DS4_TENSOR_Q8_0,
                      hc_dim, DS4_N_HC_LOWRANK, DS4_QWEN4EXP_MEM_HC));
    w->output_hc_up = qwen4exp_bind(m, plan,
            QW_SPEC2("output_hc_up.weight", DS4_TENSOR_Q8_0,
                      DS4_N_HC_LOWRANK, hc_dim, DS4_QWEN4EXP_MEM_HC));
    w->output_hc_norm = qwen4exp_bind(m, plan,
            QW_SPEC1("output_hc_norm.weight", DS4_TENSOR_F32,
                      hc_dim, DS4_QWEN4EXP_MEM_HC));

    w->output_hc_norm_offset =
        qwen4exp_norm_weight_offset(m, w->output_hc_norm, "output_hc_norm.weight");

    weights_bind_qwen4exp_ple_table(&w->ple, m, plan);

    for (uint32_t il = 0; il < w->n_layer; il++) {
        weights_bind_qwen4exp_layer(&w->layer[il], m, plan, il);
    }

    for (int f = 0; f < DS4_QWEN4EXP_MEM_COUNT; f++) {
        plan->total_bytes += plan->bytes[f];
    }
    plan->ssd_bytes = plan->bytes[DS4_QWEN4EXP_MEM_PLE];
    plan->resident_bytes = plan->total_bytes - plan->ssd_bytes;

    if (plan->bound_tensors != m->n_tensors) {
        ds4_log(stderr, DS4_LOG_WARNING,
                "ds4: qwen4exp: bound %" PRIu64 " of %" PRIu64
                " tensors in the file; %" PRIu64 " are not part of this "
                "family and stay unused\n",
                plan->bound_tensors, m->n_tensors,
                m->n_tensors - plan->bound_tensors);
    }

    qwen4exp_memory_plan_print(m, plan, &w->ple);
    qwen4exp_memory_guard(plan);
}

/*
 * Bind the multi-token-prediction head from its own GGUF.
 *
 * The head is one block at index n_layer.  It is bound as a full-attention
 * block unconditionally: the file's attention.compress_ratios carries 0 at
 * that index even though the block ships attn_q/k/v/output and indexer.*, so
 * the interval rule must not be consulted here.
 */
/* Reachable from the DS4_TEST_HOOKS entry points below and from the engine
 * open path once L5 wires it; upstream marks the same shape of function with
 * DS4_MAYBE_UNUSED (see weights_model_map_sharded_spans in ds4.c). */
static DS4_MAYBE_UNUSED void weights_bind_qwen4exp_mtp(ds4_qwen4exp_mtp_weights *w,
                                      const ds4_model *m) {
    memset(w, 0, sizeof(*w));

    const uint32_t il = DS4_N_LAYER;
    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t hc_dim = g_ds4_qwen4exp.hc_dim;
    ds4_qwen4exp_mem_plan *plan = &w->plan;
    char nm[128];

    w->block_index = il;

    /* Bind the block itself.  weights_bind_qwen4exp_layer() derives the block
     * type from the interval rule, which is wrong here, so force the QSA path
     * by binding through it with the flag already set. */
    qwen4exp_bind_block_tensors(&w->block, m, plan, il, true, false);

    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.eh_proj.weight", il);
    w->eh_proj = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            2ull * n_embd, n_embd, DS4_QWEN4EXP_MEM_DENSE));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.enorm.weight", il);
    w->enorm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            n_embd, DS4_QWEN4EXP_MEM_DENSE));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.hnorm.weight", il);
    w->hnorm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            hc_dim, DS4_QWEN4EXP_MEM_DENSE));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.hc_head_down.weight", il);
    w->hc_head_down = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            hc_dim, DS4_N_HC_LOWRANK, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.hc_head_up.weight", il);
    w->hc_head_up = qwen4exp_bind(m, plan, QW_SPEC2(nm, DS4_TENSOR_Q8_0,
            DS4_N_HC_LOWRANK, hc_dim, DS4_QWEN4EXP_MEM_HC));
    qwen4exp_name(nm, sizeof(nm), "blk.%u.nextn.hc_head_norm.weight", il);
    w->hc_head_norm = qwen4exp_bind(m, plan, QW_SPEC1(nm, DS4_TENSOR_F32,
            hc_dim, DS4_QWEN4EXP_MEM_HC));

    for (int f = 0; f < DS4_QWEN4EXP_MEM_COUNT; f++) {
        plan->total_bytes += plan->bytes[f];
    }
    plan->resident_bytes = plan->total_bytes;

    if (plan->bound_tensors != m->n_tensors) {
        qwen4exp_die("the MTP head holds %" PRIu64 " tensors but %" PRIu64
                     " were bound", m->n_tensors, plan->bound_tensors);
    }
}

#ifdef DS4_TEST_HOOKS
/* Test entry point: open, validate, bind and report, with no engine or graph.
 * Returns the plan so a test can compare it against the GGUF header bytes. */
void ds4_qwen4exp_test_open(const char *path,
                            ds4_qwen4exp_mem_plan *plan_out,
                            uint64_t *ple_rows_out,
                            uint64_t *ple_row_bytes_out,
                            uint32_t *n_shards_out,
                            uint64_t *n_tensors_out) {
    static ds4_model model;
    static ds4_qwen4exp_weights weights;

    model_open(&model, path, false, false);
    ds4_str arch = {0};
    if (!model_get_string(&model, "general.architecture", &arch) ||
        !ds4_streq(arch, "qwen4exp")) {
        qwen4exp_die("general.architecture is not qwen4exp");
    }
    config_validate_qwen4exp_model(&model);
    weights_bind_qwen4exp(&weights, &model);

    if (plan_out) *plan_out = weights.plan;
    if (ple_rows_out) *ple_rows_out = weights.ple.rows;
    if (ple_row_bytes_out) *ple_row_bytes_out = weights.ple.row_bytes;
    if (n_shards_out) *n_shards_out = model.n_shards;
    if (n_tensors_out) *n_tensors_out = model.n_tensors;

    model_close(&model);
}

/* Open, validate and bind an MTP head GGUF on its own. */
void ds4_qwen4exp_test_open_mtp(const char *path,
                                uint64_t *bound_out,
                                uint64_t *n_tensors_out,
                                uint64_t *bytes_out) {
    static ds4_model model;
    static ds4_qwen4exp_mtp_weights mtp;

    model_open(&model, path, false, false);
    ds4_str arch = {0};
    if (!model_get_string(&model, "general.architecture", &arch) ||
        !ds4_streq(arch, "qwen4exp")) {
        qwen4exp_die("general.architecture is not qwen4exp");
    }
    /* Opened on its own the head is checked against the production shape.
     * Opened after its target -- which is how the engine loads it, the target
     * first and then mtp_path -- the reduced geometry is already in place and
     * the head must agree with THAT: nextn_shared_target_tensors means the two
     * files are one model, so re-imposing the production shape here would
     * check the head against a target nobody loaded. */
    if (!ds4_model_is_qwen4exp()) g_ds4_shape = DS4_SHAPE_QWEN4EXP;
    config_validate_qwen4exp_mtp_model(&model);
    weights_bind_qwen4exp_mtp(&mtp, &model);

    if (bound_out) *bound_out = mtp.plan.bound_tensors;
    if (n_tensors_out) *n_tensors_out = model.n_tensors;
    if (bytes_out) *bytes_out = mtp.plan.total_bytes;
    model_close(&model);
}

/* Sum of every tensor's byte count straight from the GGUF headers, with no
 * family bookkeeping: the number the memory plan must reproduce. */
void ds4_qwen4exp_test_header_bytes(const char *path,
                                    uint64_t *total_out,
                                    uint64_t *ple_out) {
    static ds4_model model;
    model_open(&model, path, false, false);
    uint64_t total = 0, ple = 0;
    for (uint64_t i = 0; i < model.n_tensors; i++) {
        total += model.tensors[i].bytes;
        if (ds4_streq(model.tensors[i].name, "per_layer_token_embd.weight")) {
            ple = model.tensors[i].bytes;
        }
    }
    if (total_out) *total_out = total;
    if (ple_out) *ple_out = ple;
    model_close(&model);
}

/* Shard join order: the tensor table must be shard 0's directory, then shard
 * 1's, and so on.  Returns the map_index of each tensor in table order. */
uint64_t ds4_qwen4exp_test_shard_order(const char *path,
                                       uint32_t *out,
                                       uint64_t cap) {
    static ds4_model model;
    model_open(&model, path, false, false);
    const uint64_t n = model.n_tensors < cap ? model.n_tensors : cap;
    for (uint64_t i = 0; i < n; i++) out[i] = model.tensors[i].map_index;
    const uint64_t total = model.n_tensors;
    model_close(&model);
    return total;
}
#endif /* DS4_TEST_HOOKS */
