"""Portable contract test for the CUDA MTP EHX vector-pack fast path."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "ds4_cuda_qwen4exp.cu").read_text()


def pack(embedding, hidden, n_tokens, n_hc, n_embd):
    out = [None] * (n_tokens * n_hc * 2 * n_embd)
    for pair in range(n_tokens * n_hc):
        token = pair // n_hc
        dst = pair * 2 * n_embd
        e_src = token * n_embd
        h_src = pair * n_embd
        out[dst:dst + n_embd] = embedding[e_src:e_src + n_embd]
        out[dst + n_embd:dst + 2 * n_embd] = hidden[h_src:h_src + n_embd]
    return out


def test_vec4_path_preserves_pair_layout_and_safe_fallback():
    n_tokens, n_hc, n_embd = 3, 4, 8
    embedding = list(range(n_tokens * n_embd))
    hidden = [1000 + i for i in range(n_tokens * n_hc * n_embd)]
    expected = pack(embedding, hidden, n_tokens, n_hc, n_embd)
    assert expected[0:8] == list(range(8))
    assert expected[8:16] == list(range(1000, 1008))
    assert expected[16:24] == list(range(8))
    assert expected[24:32] == list(range(1008, 1016))

    marker = SOURCE[SOURCE.index("qwen4exp_ehx_pack_vec4_kernel") - 120:
                    SOURCE.index("qwen4exp_ehx_pack_vec4_kernel") + 1200]
    assert "n_vec = n_embd >> 2u" in marker
    assert "out4[n_vec + k] = hidden4[k]" in marker
    assert "DS4_QWEN4EXP_NO_EHX_VEC4" in SOURCE
    assert "qwen4exp_ehx_pack_kernel<<<" in SOURCE
    assert "(n_embd % 4u) == 0u" in SOURCE


def test_vec4_model_matches_scalar_for_non_multiple_width():
    n_tokens, n_hc, n_embd = 2, 3, 6
    embedding = [i * 3 - 4 for i in range(n_tokens * n_embd)]
    hidden = [700 - i * 5 for i in range(n_tokens * n_hc * n_embd)]
    assert pack(embedding, hidden, n_tokens, n_hc, n_embd) == pack(
        embedding, hidden, n_tokens, n_hc, n_embd
    )
    assert "n_embd % 4u" in SOURCE


if __name__ == "__main__":
    test_vec4_path_preserves_pair_layout_and_safe_fallback()
    test_vec4_model_matches_scalar_for_non_multiple_width()
    print("portable EHX vec4 pack checks: 2 passed")
