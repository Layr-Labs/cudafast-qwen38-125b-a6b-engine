#!/usr/bin/env python3
"""Compare current folded GDN with uniform gates and bounded replay unrolling.

Reuses the geometry fixture's independent state oracle and rotating benchmark.
Modes: 0 legacy eager, 1 uniform gates, 2 unchanged production,
3 bounded replay, 4 both. All four folded variants use 128 threads.
Pass the same arguments as test_gdn_deferred_geometry.py except --ablation,
--split-loop and --stream-state, which are controlled here.
"""
from pathlib import Path

fixture = Path(__file__).with_name('test_gdn_deferred_geometry.py')
program = fixture.read_text()
program = program.replace('if a.ablation:a.split_loop=a.stream_state=True',
                          'a.ablation=True; a.split_loop=a.stream_state=False')
program = program.replace("'.build/gdn-ablation'", "'.build/gdn-followup'")
start = program.index('if a.ablation:\n    split=')
end = program.index("header+='\\n#define ABLATION '", start)
program = program[:start] + '''
def uniform_gate(code):
    old = """            if (lane == 0u) {
                if constexpr (Precomputed) {
                    const float2 pair = gate_pairs[gate];
                    g = pair.x; beta = pair.y;
                } else {
                    g = expf(decay_coeff * qwen4exp_gdn_softplus(raw_alpha[gate] + bias));
                    beta = qwen4exp_gdn_sigmoid(raw_beta[gate]);
                }
            }
            g = __shfl_sync(0xffffffffu, g, 0);
            beta = __shfl_sync(0xffffffffu, beta, 0);"""
    new = """            if constexpr (Precomputed) {
                const float2 pair = gate_pairs[gate];
                g = pair.x; beta = pair.y;
            } else {
                if (lane == 0u) {
                    g = expf(decay_coeff * qwen4exp_gdn_softplus(raw_alpha[gate] + bias));
                    beta = qwen4exp_gdn_sigmoid(raw_beta[gate]);
                }
                g = __shfl_sync(0xffffffffu, g, 0);
                beta = __shfl_sync(0xffffffffu, beta, 0);
            }"""
    assert code.count(old)==2
    return code.replace(old,new)
def bounded(code):
    old='    for (uint32_t step = 0; step < prefix; step++) {'
    new='    #pragma unroll\\n    for (uint32_t step = 0; step < 3u; step++) {\\n        if (step >= prefix) break;'
    assert code.count(old)==1
    return code.replace(old,new)
# Replace the already emitted candidate, then emit the two isolated arms.
combined=bounded(uniform_gate(k))
assert header.count(k)==1
header=header.replace(k,combined)
header+='\\ntemplate<bool Precomputed>\\n'+uniform_gate(k).replace('geometry_kernel','split_kernel')
header+='\\ntemplate<bool Precomputed>\\n'+bounded(k).replace('geometry_kernel','cache_kernel')
''' + program[end:]
exec(compile(program, str(fixture), 'exec'))
