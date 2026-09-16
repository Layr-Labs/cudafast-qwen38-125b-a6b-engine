#!/usr/bin/env python3
"""Run the actual down kernels under a host CUDA-lane scheduler.

This checks indexing, deferred panel copies and exact operation ordering.
It is not native GPU execution or performance evidence.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
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
        ('qwen4exp_word_aligned', False),
        ('dev_qwen4exp_group_decode', False),
        ('qwen4exp_group_accumulate', False),
        ('qwen4exp_shared_vector_accumulate', False),
        ('qwen4exp_down_panel_reuse', False),
        ('qwen4exp_moe_down_q_kernel', True),
        ('qwen4exp_moe_down_reuse_kernel', True)):
    code += definition(name, templated)
with tempfile.TemporaryDirectory(prefix='moe-down-reuse-') as tmp:
    tmp = Path(tmp)
    (tmp / 'moe_down_bodies.inc').write_text(code)
    flags = ['-std=c++17', '-O2', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas']
    if a.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    exe = tmp / 'test'
    subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                    '-I', str(root),
                    str(root / 'tests/test_moe_down_reuse_host.cpp'),
                    '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
