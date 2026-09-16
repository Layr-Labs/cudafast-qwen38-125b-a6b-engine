#!/usr/bin/env python3
"""Check actual cooperative gate/up bodies with host CUDA-lane emulation.

MMA operands, selected dot products, scaling, reductions, panel row indexing,
expert routing and output bounds are compared against the original DP4A path.
This is an offline correctness check, not GPU execution or timing evidence.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, default=root / 'ds4_cuda_qwen4exp.cu')
p.add_argument('--sanitize', action='store_true')
a = p.parse_args()
src = a.source.read_text()


def definition(name, template=False):
    at = src.index(name + '(')
    start = src.rfind('\n', 0, at) + 1
    if template:
        start = src.rfind('template <', 0, start)
    end = src.index('\n}', at) + 2
    return src[start:end] + '\n'


code = ''
for name in ('cuda_block_q4_K', 'cuda_block_q5_1',
             'cuda_block_q5_K', 'cuda_block_q6_K'):
    end = src.index('} ' + name + ';') + len(name) + 3
    code += src[src.rfind('typedef struct {', 0, end):end] + '\n'
for name, templated in (
        ('warp_sum_f32', False), ('dev_q4_K_get_scale_min', False),
        ('qwen4exp_load_i8x4', False), ('qwen4exp_dp4a', True),
        ('qwen4exp_word_aligned', False), ('qw_load_words8', False),
        ('qw_load_words6', False), ('qw_pack4', False),
        ('qw_tile_store_group', False), ('qw_tile_word', False),
                ('dev_qwen4exp_group_decode', False),
        ('dev_qwen4exp_group_decode_w', False), ('qw_raw_load', True),
        ('qwen4exp_group_accumulate', False),
        ('qwen4exp_shared_vector_accumulate', False),
        ('qw_gu_coop_raw_load', False)):
    code += definition(name, templated)
kernel = definition('qwen4exp_moe_gateup_split_kernel', True)
# Preserve the complete, unchanged promoted DP4A kernel as the oracle.
code += kernel.replace('qwen4exp_moe_gateup_split_kernel(', 'gateup_parent(')
code += definition('qwen4exp_moe_gateup_diagonal_mma_kernel')
with tempfile.TemporaryDirectory(prefix='gateup-diagonal-mma-') as tmp:
    tmp = Path(tmp)
    (tmp / 'gateup_mma_bodies.inc').write_text(code)
    flags = ['-std=c++17', '-O2', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas',
             '-Wno-unused-parameter']
    if a.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    exe = tmp / 'test'
    subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                    '-I', str(root),
                    str(root / 'tests/test_gateup_diagonal_mma_host.cpp'),
                    '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
