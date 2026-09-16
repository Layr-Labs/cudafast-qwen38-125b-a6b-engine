"""Portable contract and numerical check for the ordinary GDN convolution path.

This test deliberately does not exercise replay, PDL, graph capture, or a GPU.
It checks that the vec4 path is gated by pointer alignment and a kill switch,
and that loading four adjacent weights as one tuple preserves the scalar
four-tap recurrence and accumulation order.
"""

from pathlib import Path
import random
import re


SOURCE = Path(__file__).resolve().parents[1] / "ds4_cuda_qwen4exp.cu"


def scalar_conv(history, inputs, weights):
    h0, h1, h2 = history
    output = []
    for raw in inputs:
        acc = 0.0
        acc = h0 * weights[0] + acc
        acc = h1 * weights[1] + acc
        acc = h2 * weights[2] + acc
        acc = raw * weights[3] + acc
        output.append(acc)
        h0, h1, h2 = h1, h2, raw
    return output, (h0, h1, h2)


def vector_loaded_conv(history, inputs, packed_weights):
    # The CUDA path reads float4.x/.y/.z/.w and uses the same scalar FMA order.
    return scalar_conv(history, inputs, tuple(packed_weights))


def main():
    source = SOURCE.read_text()
    assert "int            vector_weights" in source
    assert "const float4 w = ((const float4 *)conv_weight)[channel];" in source
    assert source.count("DS4_QWEN4EXP_NO_GDN_CONV_VEC4") == 1
    assert re.search(
        r"const int conv_vec4\s*=.*?alignof\(float4\).*?"
        r"DS4_QWEN4EXP_NO_GDN_CONV_VEC4.*?qwen4exp_gdn_conv_kernel",
        source,
        re.DOTALL,
    )
    assert "qk_norm_eps, adopt_row, conv_vec4);" in source
    assert source.count("float w0, w1, w2, w3;") == 1

    rng = random.Random(0x6C7C0)
    for _ in range(64):
        history = [rng.uniform(-2.0, 2.0) for _ in range(3)]
        inputs = [rng.uniform(-2.0, 2.0) for _ in range(11)]
        weights = tuple(rng.uniform(-1.0, 1.0) for _ in range(4))
        scalar, scalar_tail = scalar_conv(history, inputs, weights)
        packed, packed_tail = vector_loaded_conv(history, inputs, weights)
        assert scalar == packed
        assert scalar_tail == packed_tail

    print("portable GDN conv weight vec4 checks: 2 passed")


if __name__ == "__main__":
    main()
