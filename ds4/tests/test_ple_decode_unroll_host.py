#!/usr/bin/env python3
"""Portable contract/equivalence gate for the CUDA PLE decode overlay."""

from __future__ import annotations

import math
import struct
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "ds4_cuda_qwen4exp.cu"


def f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def sigmoid(value: float) -> float:
    return f32(1.0 / (1.0 + math.exp(-value)))


def dynamic_step(state, hyper, gated, conv_in, weight):
    channels = len(hyper)
    old_state = list(state)
    for c in range(channels):
        acc = f32(0.0)
        for k in range(4):
            i = 3 * k
            value = old_state[i * channels + c] if i < 9 else conv_in[c]
            acc = f32(f32(value * weight[4 * c + k]) + acc)
        hyper[c] = f32(hyper[c] + f32(gated[c] + f32(acc * sigmoid(acc))))
    for j in range(9):
        state[j * channels : (j + 1) * channels] = (
            old_state[(j + 1) * channels : (j + 2) * channels]
            if j < 8 else conv_in[:]
        )


def fixed_step(state, hyper, gated, conv_in, weight):
    channels = len(hyper)
    for c in range(channels):
        acc = f32(0.0)
        acc = f32(f32(state[c] * weight[4 * c + 0]) + acc)
        acc = f32(f32(state[3 * channels + c] * weight[4 * c + 1]) + acc)
        acc = f32(f32(state[6 * channels + c] * weight[4 * c + 2]) + acc)
        acc = f32(f32(conv_in[c] * weight[4 * c + 3]) + acc)
        hyper[c] = f32(hyper[c] + f32(gated[c] + f32(acc * sigmoid(acc))))
        state[c] = state[channels + c]
        state[channels + c] = state[2 * channels + c]
        state[2 * channels + c] = state[3 * channels + c]
        state[3 * channels + c] = state[4 * channels + c]
        state[4 * channels + c] = state[5 * channels + c]
        state[5 * channels + c] = state[6 * channels + c]
        state[6 * channels + c] = state[7 * channels + c]
        state[7 * channels + c] = state[8 * channels + c]
        state[8 * channels + c] = conv_in[c]


def main() -> None:
    source = SOURCE.read_text()
    required = (
        "qwen4exp_ple_conv_decode_kernel", "rows == 1u",
        "n_snapshot_rows == 0u", "conv_kernel == 4u", "dilation == 3u",
        "state_len == 9u", "DS4_QWEN4EXP_NO_PLE_DECODE_UNROLL",
        "qwen4exp_ple_conv_kernel<<<",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise AssertionError(f"missing source contract: {missing}")

    channels = 257  # Includes a non-full launch tail.
    state = [f32((i % 23 - 11) * 0.03125) for i in range(9 * channels)]
    conv_in = [f32((i % 17 - 8) * 0.02734375) for i in range(channels)]
    weight = [f32((i % 19 - 9) * 0.013671875) for i in range(4 * channels)]
    gated = [f32((i % 13 - 6) * 0.01953125) for i in range(channels)]
    hyper = [f32((i % 29 - 14) * 0.0234375) for i in range(channels)]
    dynamic_state, fixed_state = state[:], state[:]
    dynamic_hyper, fixed_hyper = hyper[:], hyper[:]
    dynamic_step(dynamic_state, dynamic_hyper, gated, conv_in, weight)
    fixed_step(fixed_state, fixed_hyper, gated, conv_in, weight)
    if dynamic_hyper != fixed_hyper or dynamic_state != fixed_state:
        raise AssertionError("fixed decode overlay diverges from dynamic shape")
    print("PLE_DECODE_UNROLL_HOST_PASS channels=257 state_len=9 taps=4 dilation=3")


if __name__ == "__main__":
    main()
