"""Portable contract test for the routed down-combine vector fast path."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "ds4_cuda_qwen4exp.cu").read_text()


def combine(partial, selected, out_dim, n_tokens, n_total_expert):
    used = len(selected) // n_tokens
    out = [0.0] * (n_tokens * out_dim)
    for token in range(n_tokens):
        for row in range(out_dim):
            acc = 0.0
            for slot in range(used):
                pair = token * used + slot
                expert = selected[pair]
                if expert < 0 or expert >= n_total_expert:
                    continue
                acc += partial[pair][row]
            out[token * out_dim + row] = acc
    return out


def combine_vec4_model(partial, selected, out_dim, n_tokens, n_total_expert):
    used = len(selected) // n_tokens
    out = [0.0] * (n_tokens * out_dim)
    for token in range(n_tokens):
        for row in range(0, out_dim, 4):
            acc = [0.0, 0.0, 0.0, 0.0]
            for slot in range(used):
                pair = token * used + slot
                expert = selected[pair]
                if expert < 0 or expert >= n_total_expert:
                    continue
                for j in range(4):
                    acc[j] += partial[pair][row + j]
            out[token * out_dim + row:token * out_dim + row + 4] = acc
    return out


def test_vec4_combine_preserves_ascending_slot_sum():
    n_tokens, n_used, n_total, out_dim = 3, 4, 512, 12
    selected = [0, 17, -1, 511, 3, 8, 99, 512, 7, 6, 5, 4]
    partial = [
        [float((pair + 1) * (row - 5)) / 17.0 for row in range(out_dim)]
        for pair in range(n_tokens * n_used)
    ]
    assert combine_vec4_model(partial, selected, out_dim, n_tokens, n_total) == combine(
        partial, selected, out_dim, n_tokens, n_total
    )


def test_source_has_alignment_gate_fallback_and_control():
    kernel = SOURCE[SOURCE.index("qwen4exp_moe_down_combine_vec4_kernel"):]
    assert "const uint32_t n_vec = out_dim >> 2u" in kernel
    assert "acc.x += v.x" in kernel
    assert "DS4_QWEN4EXP_NO_COMBINE_VEC4" in SOURCE
    assert "(out_dim % 4u) == 0u" in SOURCE
    assert "qwen4exp_moe_down_combine_grid_kernel<<<" in SOURCE


if __name__ == "__main__":
    test_vec4_combine_preserves_ascending_slot_sum()
    test_source_has_alignment_gate_fallback_and_control()
    print("portable MoE down-combine vec4 checks: 2 passed")
