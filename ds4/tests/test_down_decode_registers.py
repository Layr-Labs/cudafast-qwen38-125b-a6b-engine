#!/usr/bin/env python3
"""Compile actual down-MMA code and check register-resident Q5 decoding."""
import argparse
from pathlib import Path
import re
import subprocess


def extract(src, name):
    match = re.search(r'^__(?:device|global)__[^;{]*?\b' + name + r'\s*\(', src, re.M)
    if not match:
        raise ValueError(name)
    start = match.start()
    prior = src[:start].rstrip()
    template = prior.rfind('\ntemplate <')
    if template >= 0 and '{' not in prior[template:]:
        start = template + 1
    end = src.index('{', match.end()) + 1
    depth = 1
    while depth:
        depth += (src[end] == '{') - (src[end] == '}')
        end += 1
    return src[start:end]


def body(src, namespace):
    names = ['dev_f16_to_f32', 'dev_q4_K_get_scale_min', 'qwen4exp_word_aligned',
             'qw_load_words8', 'qw_load_words6', 'dev_qwen4exp_group_decode',
             'qw_pack4', 'qw_tile_word', 'qw_tile_copy_group', 'qw_tile_store_group',
             'qw_tile_store_zero', 'qw_raw_load', 'dev_qwen4exp_group_decode_w',
             'qw_prefetch_l2', 'qw_mma_m16n8k32', 'qwen4exp_moe_down_mma_kernel']
    structs = re.findall(r'typedef struct \{[^}]*\} cuda_block_\w+;', src)
    return 'namespace ' + namespace + ' {\n' + '\n'.join(structs + [extract(src, n) for n in names]) + '\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--replays', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / '.build/down-decode-registers'
    output.mkdir(parents=True, exist_ok=True)
    prefix = '''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "ds4_qwen4exp_moe_types.h"
#define DEFINE_TYPE(name, id) DS4_QWEN4EXP_TY_##name = id,
enum { DS4_QWEN4EXP_MOE_TYPES(DEFINE_TYPE) };
#undef DEFINE_TYPE
#define CUDA_QK_K 256
#define DS4_QWEN4EXP_WIDE_PAYLOAD 1
#define QW_MMA_BN 32
#define QW_MMA_G 4
#define QW_MMA_LD 132
#define QW_DOWN_MMA_BM 64
#define QW_DOWN_MMA_THREADS 128
#define QW_DOWN_MMA_NT 4
'''
    code = prefix + body(args.baseline.read_text(), 'baseline')
    code += body((root / 'ds4/ds4_cuda_qwen4exp.cu').read_text(), 'candidate')
    (output / 'kernels.cuh').write_text(code)
    exe = output / 'test'
    subprocess.run(['nvcc', '-O3', '-std=c++17', '-arch=' + args.arch,
                    '-ftz=false', '-prec-div=true', '-prec-sqrt=true',
                    '-I' + str(output), '-I' + str(root / 'ds4'),
                    str(Path(__file__).with_suffix('.cu')), '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(args.samples), str(args.replays)], check=True)


if __name__ == '__main__':
    main()
