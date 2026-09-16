"""Portable guard and arithmetic check for the CUDA indexer decode variant.

This does not run CUDA.  It models the old 128-element shared tree and the
new four-warp shuffle reduction with float32 rounding, and checks that the
score delta is bounded on deterministic finite inputs.  GB10 correctness and
top-k behavior still require the organizer's native oracle.
"""
from pathlib import Path
import math
import random
import struct


SOURCE = (Path(__file__).resolve().parents[1] / "ds4_cuda.cu").read_text()
assert "glm_indexer_scores_decode_warp_kernel" in SOURCE
assert "__shfl_down_sync" in SOURCE
assert ("g_quality_mode && n_tokens == 1u && n_head == 4u &&" in SOURCE)
assert "DS4_CUDA_NO_INDEXER_WARP_REDUCE" in SOURCE
assert "glm indexer scores decode warp launch" in SOURCE


def f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def add(a, b):
    return f32(f32(a) + f32(b))


def old_tree(values):
    p = [f32(v) for v in values]
    stride = 64
    while stride:
        for i in range(stride):
            p[i] = add(p[i], p[i + stride])
        stride >>= 1
    return p[0]


def shuffle_tree(values):
    partial = []
    for base in range(0, 128, 32):
        p = [f32(v) for v in values[base:base + 32]]
        stride = 16
        while stride:
            for i in range(32 - stride):
                if i < stride:
                    p[i] = add(p[i], p[i + stride])
            stride >>= 1
        partial.append(p[0])
    return add(add(partial[0], partial[1]), add(partial[2], partial[3]))


rng = random.Random(340449)
max_delta = 0.0
for case in range(256):
    if case == 0:
        values = [0.0] * 128
    elif case == 1:
        values = [1.0 if i & 1 else -1.0 for i in range(128)]
    elif case == 2:
        values = [1.0e20 if i % 3 else -1.0e20 for i in range(128)]
    else:
        values = [rng.uniform(-3.0, 3.0) for _ in range(128)]
    a = old_tree(values)
    b = shuffle_tree(values)
    assert math.isfinite(a) and math.isfinite(b)
    max_delta = max(max_delta, abs(a - b))
    assert abs(a - b) <= 2.0e-5 * max(1.0, abs(a), abs(b))

print(f"indexer warp reduction host model: PASS (max dot delta={max_delta:.8g})")
