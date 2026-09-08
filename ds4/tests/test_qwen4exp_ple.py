#!/usr/bin/env python3
"""Exactness tests for the host side of the Qwen4-Exp PLE n-gram table.

The C module under test is ds4_qwen4exp_ple.c, driven through the probe modes
of tests/test_qwen4exp_ple.  This file holds three things the C side must agree
with, written independently here:

  * the n-gram hash of the mlx reference (Qwen4ExpNGram.swift), in its shift
    and segment form rather than the carried-history recurrence the C uses;
  * ggml's dequantize_row_iq4_nl, from the block layout;
  * a GGUF writer, so a synthetic table of a few million rows can be built
    without the 26.8 GiB checkpoint.

Run: python3 tests/test_qwen4exp_ple.py [--rows N] [--keep]
"""

import argparse
import hashlib
import os
import random
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "test_qwen4exp_ple")

# GGUF metadata value types.
T_UINT32 = 4
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10

GGUF_TYPE_IQ4_NL = 20
IQ4_NL_BLOCK_ELEMS = 32
IQ4_NL_BLOCK_BYTES = 18
ALIGNMENT = 32

# The multipliers of the pinned checkpoint (qwen4exp.ple.layer_multipliers).
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]

KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                 1, 13, 25, 38, 53, 69, 89, 113]

# The metadata shard of the pinned checkpoint: 10.9 MB, no tensors, every
# qwen4exp.ple.* key.  Pinned by revision and by content, so a re-tagged or
# re-quantized upload cannot silently change what this test asserts.
REAL_SHARD_URL = (
    "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/"
    "38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
REAL_SHARD_SHA256 = \
    "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"
REAL_SHARD_BYTES = 10946624

# What the shard must say.  Written out here rather than read from the file,
# so a key the reader spells wrongly shows up as a missing value.
REAL_CONSTANTS = {
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "head_count": 16,
    "row_dim": 160,
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "conv_kernel": 4,
    "ple_layers": [1],
    "multipliers": [23703573157769, 20109073645365, 8052911324071],
    "head_vocab_sizes": [
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
        20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
        20000161, 20000171],
    "head_offsets": [
        0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
        140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
        260000955, 280001114, 300001275],
    "row_total": 320001446,
}

# The mlx reference builds the constants rather than storing them; these are
# its two formulas (Qwen4ExpNGramConstants in Qwen4ExpNGram.swift).
NGRAM_VOCAB_SIZE_BASE = 20000000
CONFIG_SEED = 1234
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1

FAILED = 0
TOTAL = 0


def check(cond, msg):
    global FAILED, TOTAL
    TOTAL += 1
    if not cond:
        FAILED += 1
        print("  FAIL: %s" % msg)


def section(name):
    print("RUN: %s" % name)


# --------------------------------------------------------------------------
# Reference: the n-gram hash of Qwen4ExpNGram.swift.
#
# The reference shifts the whole sequence and blanks a shift that crosses an
# end-of-sequence token or the start of the sequence.  This mirrors that shape:
# it computes, per position, how many tokens have passed since the last
# end-of-sequence token, and refuses a shift longer than that.
# --------------------------------------------------------------------------

def ref_shift_right(tokens, shift, eos):
    if shift == 0:
        return list(tokens)
    previous = -1          # index of the last end-of-sequence token before t
    in_segment = []
    for t, tok in enumerate(tokens):
        in_segment.append(t - (previous + 1))
        if tok == eos:
            previous = t
    out = []
    for t in range(len(tokens)):
        source = t - shift
        usable = in_segment[t] >= shift and source >= 0
        out.append(tokens[source] if usable else eos)
    return out


def ref_row_ids(tokens, consts):
    ngram = consts["ngram_size"]
    hpn = consts["heads_per_ngram"]
    eos = consts["eos_token_id"]
    mult = consts["multipliers"]
    sizes = consts["head_vocab_sizes"]
    offsets = consts["head_offsets"]

    shifted = [ref_shift_right(tokens, p, eos) for p in range(ngram)]
    out = [[0] * (hpn * (ngram - 1)) for _ in tokens]

    for n in range(2, ngram + 1):
        low = (n - 2) * hpn
        for t in range(len(tokens)):
            mixed = shifted[0][t] * mult[0]
            for p in range(1, n):
                mixed ^= shifted[p][t] * mult[p]
            for k in range(hpn):
                head = low + k
                out[t][head] = mixed % sizes[head] + offsets[head]
    return out


# --------------------------------------------------------------------------
# Reference: ggml dequantize_row_iq4_nl (ggml/src/ggml-quants.c).
# --------------------------------------------------------------------------

def ref_dequant_iq4_nl(blob):
    assert len(blob) % IQ4_NL_BLOCK_BYTES == 0
    out = []
    for b in range(len(blob) // IQ4_NL_BLOCK_BYTES):
        block = blob[b * IQ4_NL_BLOCK_BYTES:(b + 1) * IQ4_NL_BLOCK_BYTES]
        d = struct.unpack("<e", block[0:2])[0]
        qs = block[2:]
        values = [0.0] * IQ4_NL_BLOCK_ELEMS
        for j in range(IQ4_NL_BLOCK_ELEMS // 2):
            values[j] = d * KVALUES_IQ4NL[qs[j] & 0x0F]
            values[j + 16] = d * KVALUES_IQ4NL[qs[j] >> 4]
        out.extend(values)
    # Round every value to float32, which is what the C produces.
    return [struct.unpack("<f", struct.pack("<f", v))[0] for v in out]


def f32_bits(values):
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


# --------------------------------------------------------------------------
# A minimal GGUF writer.
# --------------------------------------------------------------------------

def _u32(v):
    return struct.pack("<I", v)


def _u64(v):
    return struct.pack("<Q", v)


def _gstr(s):
    raw = s.encode("utf-8")
    return _u64(len(raw)) + raw


def _kv(key, type_id, payload):
    return _gstr(key) + _u32(type_id) + payload


def kv_u32(key, value):
    return _kv(key, T_UINT32, _u32(value))


def kv_u64_array(key, values):
    body = _u32(T_UINT64) + _u64(len(values)) + b"".join(_u64(v) for v in values)
    return _kv(key, T_ARRAY, body)


def kv_string_array(key, values):
    body = _u32(T_STRING) + _u64(len(values)) + b"".join(_gstr(v) for v in values)
    return _kv(key, T_ARRAY, body)


def write_gguf(path, kvs, tensors):
    """tensors: list of (name, dims, type_id, payload bytes)."""
    infos = b""
    blobs = []
    offset = 0
    for name, dims, type_id, payload in tensors:
        infos += _gstr(name) + _u32(len(dims))
        for d in dims:
            infos += _u64(d)
        infos += _u32(type_id) + _u64(offset)
        blobs.append(payload)
        size = len(payload)
        offset += (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    header = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(kvs))
    body = header + b"".join(kvs) + infos
    pad = (-len(body)) % ALIGNMENT

    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(b"\0" * pad)
        for payload in blobs:
            fh.write(payload)
            fh.write(b"\0" * ((-len(payload)) % ALIGNMENT))
    return len(body) + pad     # start of the tensor data


# --------------------------------------------------------------------------
# The synthetic checkpoint.
# --------------------------------------------------------------------------

def build_checkpoint(directory, row_count, row_dim=160, vocab=4096, drop_key=None):
    """Two shards, as the published split has them: metadata in the first,
    the table tensor in the second."""
    heads = 16
    # Sixteen distinct head slices that fit inside the table.
    per_head = row_count // (heads + 1)
    sizes = [per_head - (heads - h) for h in range(heads)]
    offsets = []
    running = 0
    for s in sizes:
        offsets.append(running)
        running += s

    consts = {
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "head_count": heads,
        "row_dim": row_dim,
        "vocab_size": vocab,
        "eos_token_id": vocab - 96,
        "multipliers": MULTIPLIERS,
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "row_total": running,
        "table_rows": row_count,
    }

    kvs = [
        (_kv("general.architecture", T_STRING, _gstr("qwen4exp")), "general.architecture"),
        (kv_u32("qwen4exp.ple.ngram_size", 3), "qwen4exp.ple.ngram_size"),
        (kv_u32("qwen4exp.ple.heads_per_ngram", 8), "qwen4exp.ple.heads_per_ngram"),
        (kv_u32("qwen4exp.embedding_length_per_layer_input", row_dim),
         "qwen4exp.embedding_length_per_layer_input"),
        (kv_u32("qwen4exp.ple.eos_token_id", consts["eos_token_id"]),
         "qwen4exp.ple.eos_token_id"),
        (kv_u32("qwen4exp.ple.conv_kernel", 4), "qwen4exp.ple.conv_kernel"),
        (kv_u64_array("qwen4exp.ple.layers", [1]), "qwen4exp.ple.layers"),
        (kv_u64_array("qwen4exp.ple.layer_multipliers", MULTIPLIERS),
         "qwen4exp.ple.layer_multipliers"),
        (kv_u64_array("qwen4exp.ple.head_vocab_sizes", sizes),
         "qwen4exp.ple.head_vocab_sizes"),
        (kv_u64_array("qwen4exp.ple.head_offsets", offsets),
         "qwen4exp.ple.head_offsets"),
        (kv_string_array("tokenizer.ggml.tokens", ["t%d" % i for i in range(vocab)]),
         "tokenizer.ggml.tokens"),
    ]
    meta = [blob for blob, name in kvs if name != drop_key]

    meta_path = os.path.join(directory, "synthetic-00001-of-00002.gguf")
    data_path = os.path.join(directory, "synthetic-00002-of-00002.gguf")
    write_gguf(meta_path, meta, [])

    row_bytes = row_dim // IQ4_NL_BLOCK_ELEMS * IQ4_NL_BLOCK_BYTES
    rng = random.Random(0xC0FFEE)
    scale_rng = random.Random(0x5CA1E5)
    # In chunks: random.randbytes() asks getrandbits() for the whole length at
    # once, and CPython caps that at 2**31 - 1 BITS, so a single call for the
    # default 3,000,000 rows (270 MB, 2.16e9 bits) raises OverflowError there.
    #
    # This does NOT claim the chunked stream equals the single call's.
    # getrandbits() draws whole 32-bit words and discards the excess bits of
    # the last one, so a chunk boundary can move the stream; it happens to
    # match on CPython 3.14 for these sizes, which is not something to rely on.
    # What the test needs is only that the bytes are DETERMINISTIC for a fixed
    # seed, chunk size and interpreter, because the reference below is computed
    # from the same bytes the engine reads.
    total_bytes = row_count * row_bytes
    chunk = 1 << 20
    payload = bytearray()
    remaining = total_bytes
    while remaining > 0:
        take = chunk if remaining > chunk else remaining
        payload += rng.randbytes(take)
        remaining -= take
    # EVERY block scale is drawn as a real fp16 value, never as random bytes.
    #
    # An IQ4_NL block is [fp16 scale][16 packed nibbles], and random bytes put
    # exponent 0x1F into roughly one scale in 32 -- a NaN or an infinity, which
    # is not a weight any checkpoint can contain.  The engine and this
    # reference then disagree on the PAYLOAD rather than on the arithmetic:
    # ds4_qwen4exp_ple.c propagates the NaN payload ggml-style (0x7eac stays
    # 0x7FD58000) while struct.unpack("<e") canonicalizes to 0x7FC00000.  Both
    # are defensible for a value that cannot occur; the fixture was the thing
    # at fault.  Same rule as the synthetic GGUF writer, which packs every
    # scale from a float (tools/qwen4exp_synthetic_gguf.py, `scale`).
    blocks_per_row = row_dim // IQ4_NL_BLOCK_ELEMS
    for row in range(row_count):
        base = row * row_bytes
        for blk in range(blocks_per_row):
            at = base + blk * IQ4_NL_BLOCK_BYTES
            if row < 4:
                # The first rows keep exactly representable scales so a
                # comparison failure is easy to read.
                value = 0.5 * (blk + 1)
            else:
                value = scale_rng.uniform(-2.0, 2.0)
            payload[at:at + 2] = struct.pack("<e", value)

    # The fixture must contain no NaN or infinite scale at all: assert it here
    # rather than discover it as a dequant mismatch a thousand rows in.
    for row in range(row_count):
        base = row * row_bytes
        for blk in range(blocks_per_row):
            at = base + blk * IQ4_NL_BLOCK_BYTES
            half = int.from_bytes(payload[at:at + 2], "little")
            if (half & 0x7C00) == 0x7C00:
                raise SystemExit(
                    "fixture defect: row %d block %d has fp16 scale 0x%04x, "
                    "exponent 0x1F (NaN or infinity)" % (row, blk, half))

    data_start = write_gguf(
        data_path, [kv_u32("split.no", 1)],
        [("per_layer_token_embd.weight", [row_dim, row_count],
          GGUF_TYPE_IQ4_NL, bytes(payload))])

    return consts, meta_path, data_path, data_start, row_bytes


# --------------------------------------------------------------------------
# Probe driver.
# --------------------------------------------------------------------------

def probe(*args):
    result = subprocess.run([PROBE] + [str(a) for a in args],
                            capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def probe_ok(*args):
    rc, out, err = probe(*args)
    if rc != 0:
        raise AssertionError("probe %s failed: %s" % (args[0], err.strip()))
    return out


def parse_stats(text):
    stats = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        stats[key] = int(value)
    return stats


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------

def test_constants(shards, consts):
    section("constants come off the checkpoint")
    out = probe_ok("constants", *shards)
    got = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        got[key] = value

    check(int(got["ngram_size"]) == consts["ngram_size"], "ngram_size")
    check(int(got["heads_per_ngram"]) == consts["heads_per_ngram"], "heads_per_ngram")
    check(int(got["head_count"]) == consts["head_count"], "head_count")
    check(int(got["row_dim"]) == consts["row_dim"], "row_dim")
    check(int(got["vocab_size"]) == consts["vocab_size"], "vocab_size")
    check(int(got["eos_token_id"]) == consts["eos_token_id"], "eos_token_id")
    check(int(got["row_total"]) == consts["row_total"], "row_total")
    for i, m in enumerate(consts["multipliers"]):
        check(int(got["multiplier[%d]" % i]) == m, "multiplier %d" % i)
    for h in range(consts["head_count"]):
        want = "%d,%d" % (consts["head_vocab_sizes"][h], consts["head_offsets"][h])
        check(got["head[%d]" % h] == want, "head %d" % h)


def test_missing_key_is_a_refusal(directory, row_count):
    section("a missing key is a refusal")
    for key in ("qwen4exp.ple.layer_multipliers",
                "qwen4exp.ple.head_offsets",
                "qwen4exp.ple.eos_token_id",
                "qwen4exp.ple.conv_kernel",
                "qwen4exp.ple.layers",
                "tokenizer.ggml.tokens"):
        sub = os.path.join(directory, "drop-" + key)
        os.makedirs(sub, exist_ok=True)
        _, meta, data, _, _ = build_checkpoint(sub, 4096, drop_key=key)
        rc, _, err = probe("constants", meta, data)
        check(rc != 0, "refuses without %s" % key)
        check(key in err, "the refusal names %s" % key)
        os.remove(meta)
        os.remove(data)


def test_row_ids(shards, consts):
    section("row ids are bit exact against the reference hash")
    eos = consts["eos_token_id"]
    vocab = consts["vocab_size"]
    rng = random.Random(1234)

    cases = [
        [eos],
        [eos, eos, eos],
        [1, eos, 2, 3],
        [1, 2, eos, eos, 3, 4, 5],
        [vocab - 1, 0, eos, vocab - 1],
    ]
    for _ in range(12):
        n = rng.randint(1, 200)
        cases.append([eos if rng.random() < 0.12 else rng.randrange(vocab)
                      for _ in range(n)])

    for tokens in cases:
        out = probe_ok("ids", ",".join(str(t) for t in tokens), *shards)
        got = [[int(v) for v in line.split()] for line in out.splitlines()]
        want = ref_row_ids(tokens, consts)
        check(got == want, "ids for a %d-token sequence" % len(tokens))
        if got != want:
            for t, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    print("    first mismatch at token %d: %r vs %r" % (t, g, w))
                    break

    section("prefill and decode carry the same history")
    tokens = [eos if rng.random() < 0.1 else rng.randrange(vocab) for _ in range(64)]
    whole = probe_ok("ids", ",".join(str(t) for t in tokens), *shards)
    check([[int(v) for v in line.split()] for line in whole.splitlines()]
          == ref_row_ids(tokens, consts), "a 64-token sequence")


def test_dequant_and_gather(shards, consts, data_path, data_start, row_bytes):
    section("gathered rows match the file bytes and the ggml reference")
    rng = random.Random(99)
    rows = sorted({rng.randrange(consts["table_rows"]) for _ in range(24)} |
                  {0, 1, 2, 3, consts["table_rows"] - 1})
    csv = ",".join(str(r) for r in rows)

    raw = probe_ok("raw", csv, *shards).splitlines()
    check(len(raw) == len(rows), "one raw row per id")

    with open(data_path, "rb") as fh:
        for i, row in enumerate(rows):
            fh.seek(data_start + row * row_bytes)
            direct = fh.read(row_bytes)
            check(raw[i] == direct.hex(), "row %d matches a direct file read" % row)

    for cache_bytes in (0, 4096, 1 << 20):
        out = probe_ok("rows", csv, cache_bytes, *shards).splitlines()
        check(len(out) == len(rows), "one dequantized row per id")
        with open(data_path, "rb") as fh:
            for i, row in enumerate(rows):
                fh.seek(data_start + row * row_bytes)
                want = f32_bits(ref_dequant_iq4_nl(fh.read(row_bytes)))
                got = [int(v, 16) for v in out[i].split()]
                check(got == want,
                      "row %d dequantizes bit exactly (cache %d)" % (row, cache_bytes))


def test_hot_set(shards, consts):
    section("the hot set counts hits, misses and evictions")
    row_bytes = consts["row_dim"] * 4

    # A working set that fits: no eviction, and every repeat is a hit.
    fits = (row_bytes + 28) * 4096
    stats = parse_stats(probe_ok("soak", 4000, 7, 64, fits, *shards))
    check(stats["evictions"] == 0, "a working set that fits evicts nothing")
    check(stats["misses"] == 64, "each of the 64 rows misses once")
    check(stats["hits"] == 4000 - 64, "every repeat is a hit")
    check(stats["resident_rows"] == 64, "64 rows stay resident")

    # No hot set at all: every row is a miss.
    stats = parse_stats(probe_ok("soak", 4000, 7, 64, 0, *shards))
    check(stats["capacity_rows"] == 0, "a zero ceiling turns the hot set off")
    check(stats["hits"] == 0 and stats["misses"] == 4000, "every row is a miss")

    # A working set that does not fit: eviction happens.
    stats = parse_stats(probe_ok("soak", 20000, 11, 100000, 1 << 18, *shards))
    check(stats["evictions"] > 0, "a working set that overflows evicts")
    check(stats["resident_rows"] == stats["capacity_rows"], "the arena fills")


def test_ceiling_and_exactness(shards, consts):
    section("the ceiling holds over a 100k-row workload")
    reference = None
    for ceiling in (0, 1 << 16, 1 << 20, 12 << 20):
        stats = parse_stats(probe_ok("soak", 100000, 20260901, 250000, ceiling, *shards))
        check(stats["resident_bytes"] <= ceiling, "resident %d <= ceiling %d"
              % (stats["resident_bytes"], ceiling))
        check(stats["hits"] + stats["misses"] == 100000, "every row is accounted for")
        if reference is None:
            reference = stats["checksum"]
        check(stats["checksum"] == reference,
              "the ceiling never changes a value (ceiling %d)" % ceiling)


def splitmix64(value):
    v = (value + SPLITMIX_GAMMA) & MASK64
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK64
    return v ^ (v >> 31)


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def primes_after(start, count):
    out = []
    p = start
    while len(out) < count:
        p += 1
        if is_prime(p):
            out.append(p)
    return out


def fixture_path():
    directory = os.environ.get(
        "DS4_PLE_FIXTURE_DIR",
        os.path.join(tempfile.gettempdir(), "ds4-ple-fixtures"))
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


def fetch_real_shard():
    """Return (path, None) or (None, reason).  Cached; never re-downloaded."""
    path = fixture_path()
    if os.path.exists(path) and os.path.getsize(path) == REAL_SHARD_BYTES:
        if hashlib.sha256(open(path, "rb").read()).hexdigest() == REAL_SHARD_SHA256:
            return path, None
        os.remove(path)

    try:
        with urllib.request.urlopen(REAL_SHARD_URL, timeout=120) as response:
            blob = response.read()
    except (urllib.error.URLError, OSError) as exc:
        return None, "the metadata shard could not be fetched (%s)" % exc

    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) != REAL_SHARD_BYTES or digest != REAL_SHARD_SHA256:
        return None, ("the fetched shard is %d bytes with sha256 %s; %d bytes and %s "
                      "were pinned" % (len(blob), digest, REAL_SHARD_BYTES,
                                       REAL_SHARD_SHA256))
    with open(path, "wb") as fh:
        fh.write(blob)
    return path, None


def deterministic_tokens(count, vocab, eos, seed):
    """A fixed sequence, generated without random(), so the same tokens are
    used on every Python build.  About one token in nine is the
    end-of-sequence token, so runs of two and three boundaries occur."""
    tokens = []
    state = seed
    for _ in range(count):
        state = splitmix64(state)
        tokens.append(eos if (state >> 40) % 9 == 0 else state % vocab)
    return tokens


def read_probe_constants(out):
    got = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        got[key] = value
    consts = {
        "ngram_size": int(got["ngram_size"]),
        "heads_per_ngram": int(got["heads_per_ngram"]),
        "head_count": int(got["head_count"]),
        "row_dim": int(got["row_dim"]),
        "vocab_size": int(got["vocab_size"]),
        "eos_token_id": int(got["eos_token_id"]),
        "conv_kernel": int(got["conv_kernel"]),
        "row_total": int(got["row_total"]),
    }
    consts["multipliers"] = [int(got["multiplier[%d]" % i])
                             for i in range(consts["ngram_size"])]
    consts["head_vocab_sizes"] = []
    consts["head_offsets"] = []
    for h in range(consts["head_count"]):
        size, offset = got["head[%d]" % h].split(",")
        consts["head_vocab_sizes"].append(int(size))
        consts["head_offsets"].append(int(offset))
    consts["ple_layers"] = []
    i = 0
    while ("ple_layer[%d]" % i) in got:
        consts["ple_layers"].append(int(got["ple_layer[%d]" % i]))
        i += 1
    return consts


def test_real_metadata_shard():
    """The synthetic fixture writes the key names the reader looks for, so it
    cannot show that those names are the ones the checkpoint uses.  This reads
    the published metadata shard instead."""
    section("the published metadata shard parses with the shipped reader")
    path, reason = fetch_real_shard()
    if path is None:
        print("  SKIP: %s" % reason)
        return

    rc, out, err = probe("constants", path)
    check(rc == 0, "the reader accepts the published shard")
    if rc != 0:
        print("    %s" % err.strip())
        return

    got = read_probe_constants(out)
    for key, want in REAL_CONSTANTS.items():
        check(got.get(key) == want,
              "%s is %r, not the published %r" % (key, got.get(key), want))

    section("the published constants re-derive from the mlx formulas")
    # Head vocab sizes are consecutive primes above the configured base.
    want_sizes = primes_after(NGRAM_VOCAB_SIZE_BASE - 1, got["head_count"])
    check(got["head_vocab_sizes"] == want_sizes,
          "the head vocab sizes are the first %d primes above %d"
          % (got["head_count"], NGRAM_VOCAB_SIZE_BASE - 1))
    running = 0
    want_offsets = []
    for size in want_sizes:
        want_offsets.append(running)
        running += size
    check(got["head_offsets"] == want_offsets, "the head offsets accumulate")
    check(got["row_total"] == running, "the head table covers %d rows" % running)

    # Multipliers come off a splitmix64 stream seeded by the configuration
    # seed and the ordinal of the PLE layer, which is 0 for the only one.
    half = max(1, (INT64_MAX // got["vocab_size"]) // 2)
    base = (CONFIG_SEED + 10007 * 0) & MASK64
    want_multipliers = [
        2 * (splitmix64((base + SPLITMIX_GAMMA * (i + 1)) & MASK64) % half) + 1
        for i in range(got["ngram_size"])]
    check(got["multipliers"] == want_multipliers,
          "the multipliers come off splitmix64 seeded with %d" % CONFIG_SEED)

    section("row ids over a 1088-token sequence against the published keys")
    tokens = deterministic_tokens(1088, got["vocab_size"], got["eos_token_id"],
                                  0x5EED1088)
    check(tokens.count(got["eos_token_id"]) > 40,
          "the sequence crosses many end-of-sequence boundaries")
    ids = probe_ok("ids", ",".join(str(t) for t in tokens), path)
    mine = [[int(v) for v in line.split()] for line in ids.splitlines()]
    want = ref_row_ids(tokens, got)
    check(mine == want, "1088 tokens of ids are bit exact")
    if mine != want:
        for t, (a, b) in enumerate(zip(mine, want)):
            if a != b:
                print("    first mismatch at token %d: %r vs %r" % (t, a, b))
                break

    # Every product the hash forms must stay non-negative in signed 64-bit,
    # which is what lets the reader use an unsigned remainder.
    biggest = (got["vocab_size"] - 1) * max(got["multipliers"])
    check(biggest <= INT64_MAX, "no token times a multiplier overflows int64")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000000,
                        help="rows in the synthetic table")
    parser.add_argument("--keep", action="store_true",
                        help="keep the synthetic checkpoint")
    args = parser.parse_args()

    if not os.access(PROBE, os.X_OK):
        print("error: %s is not built; run make tests/test_qwen4exp_ple" % PROBE)
        return 2

    directory = tempfile.mkdtemp(prefix="ds4-ple-")
    try:
        consts, meta, data, data_start, row_bytes = build_checkpoint(directory, args.rows)
        shards = [meta, data]
        print("synthetic table: %d rows, %.1f MiB"
              % (args.rows, args.rows * row_bytes / (1 << 20)))

        test_constants(shards, consts)
        test_missing_key_is_a_refusal(directory, args.rows)
        test_row_ids(shards, consts)
        test_dequant_and_gather(shards, consts, data, data_start, row_bytes)
        test_hot_set(shards, consts)
        test_ceiling_and_exactness(shards, consts)
        test_real_metadata_shard()
    finally:
        if not args.keep:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory)
        else:
            print("kept %s" % directory)

    print("%d/%d checks passed" % (TOTAL - FAILED, TOTAL))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
