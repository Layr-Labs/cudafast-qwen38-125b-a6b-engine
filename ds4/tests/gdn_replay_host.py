#!/usr/bin/env python3
"""Exercise actual CUDA recurrence bodies and graph lifecycle code on the host.

No weights or GPU are needed. This is a structural/rounding-order check, not
native CUDA arithmetic, device execution, or a performance measurement.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=root / 'ds4_cuda_qwen4exp.cu')
p.add_argument('--graph', type=Path, default=root / 'ds4_qwen4exp_graph.inc')
p.add_argument('--sanitize', action='store_true')
p.add_argument('--controller-only', action='store_true')
a = p.parse_args()
src = a.source.read_text()

def definition(source, name, template=False):
    at = source.index(name + '(')
    start = source.rfind('\n', 0, at) + 1
    if template:
        start = source.rfind('\n', 0, start - 1) + 1
    end = source.index('\n}', at) + 2
    return source[start:end] + '\n'

code = ''.join(definition(src, n) for n in (
    'warp_sum_all_f32', 'dot4_f32', 'qwen4exp_gdn_sigmoid', 'qwen4exp_gdn_softplus'))
code += definition(src, 'qwen4exp_gdn_recurrence_kernel', True)
code += definition(src, 'qwen4exp_gdn_replay_kernel')
with tempfile.TemporaryDirectory(prefix='gdn-replay-host-') as tmp:
    tmp = Path(tmp)
    (tmp / 'gdn_replay_bodies.inc').write_text(code)
    flags = ['-O2', '-std=c++17', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas']
    if a.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    tests = [] if a.controller_only else ['test_gdn_replay_host']
    controller = root / 'tests/test_gdn_replay_controller.cpp'
    if controller.exists():
        graph = a.graph.read_text()
        # copy_layers also has a forward declaration; take its definition.
        first = graph.index('static bool qwen4exp_session_copy_layers(')
        body = graph.index('static bool qwen4exp_session_copy_layers(', first + 1)
        ctl = definition(graph, 'qwen4exp_gdn_replay_snapshot') + definition(
            graph[body:], 'qwen4exp_session_copy_layers') + ''.join(
            definition(graph, n) for n in (
                'qwen4exp_session_select_layers', 'qwen4exp_session_settle_layers',
                'ds4_qwen4exp_session_reset'))
        start = graph.index('    const ds4_qwen4exp_gdn_replay_step replay_step =')
        end = graph.index('    if (s->d_pos &&', start)
        ctl += ('static bool prepare(ds4_qwen4exp_session *s, uint32_t n_tokens) {\n'
                'const ds4_qwen4exp_config *cfg = &g_ds4_qwen4exp;\n' +
                graph[start:end] + '\nreturn true;\n}\n')
        (tmp / 'gdn_replay_controller.inc').write_text(ctl)
        tests.append('test_gdn_replay_controller')
    for name in tests:
        subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                        '-I', str(root), str(root / 'tests' / (name + '.cpp')),
                        '-o', str(tmp / name)], check=True)
        subprocess.run([str(tmp / name)], check=True)
