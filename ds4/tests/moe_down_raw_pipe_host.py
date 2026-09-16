#!/usr/bin/env python3
"""Execute direct and pipelined down-MMA bodies using host CUDA fibers.

Compare full partial buffers, guards, empty/compact expert lists, tile tails,
and signed-zero/subnormal cases. This is not GPU timing or execution.
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
p.add_argument('--header', type=Path, default=root / 'ds4_cuda_down_raw_pipe.cuh')
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
start = src.index('#define QW_MMA_BM 32')
end = src.index('/* Ceiling on the decode down', start)
code += src[start:end]
for name in ('cuda_block_q4_K', 'cuda_block_q5_1',
             'cuda_block_q5_K', 'cuda_block_q6_K'):
    end = src.index('} ' + name + ';') + len(name) + 3
    code += src[src.rfind('typedef struct {', 0, end):end] + '\n'
for name, templated in (
        ('dev_q4_K_get_scale_min', False),
        ('qwen4exp_load_i8x4', False), ('qwen4exp_dp4a', True),
        ('qwen4exp_word_aligned', False), ('qw_load_words8', False),
        ('qw_load_words6', False), ('qw_pack4', False),
        ('qw_tile_store_group', False), ('qw_tile_word', False),
        ('qw_tile_store_zero', False), ('qw_tile_copy_group', False),
                ('dev_qwen4exp_group_decode', False),
        ('dev_qwen4exp_group_decode_w', False), ('qw_raw_load', True)):
    code += definition(name, templated)
code += definition('qwen4exp_moe_down_mma_kernel', True)
code += a.header.read_text().replace('#pragma once', '')
with tempfile.TemporaryDirectory(prefix='moe-down-raw-pipe-') as tmp:
    tmp = Path(tmp)
    (tmp / 'down_raw_pipe_bodies.inc').write_text(code)
    flags = ['-std=c++17', '-O2', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas',
             '-Wno-unused-parameter']
    if a.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    exe = tmp / 'test'
    subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                    '-I', str(root),
                    str(root / 'tests/test_moe_down_raw_pipe_host.cpp'),
                    '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
