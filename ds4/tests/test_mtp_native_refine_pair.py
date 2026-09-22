#!/usr/bin/env python3
"""Compile actual production kernels and compare paired refinement bit for bit.

Synthetic Q8_0 data only; no checkpoint or benchmark score. Requires CUDA GPU.
"""
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[2]
build = root / '.build' / 'refine-pair-test'
build.mkdir(parents=True, exist_ok=True)
source = (root / 'ds4/ds4_cuda.cu').read_text()
def function(name):
    start = source.rfind('__device__', 0, source.index(name + '('))
    begin = source.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]
header = (root / 'ds4/ds4_cuda_mtp_native.cuh').read_text()
(build / 'production.cuh').write_text('\n'.join(function(n) for n in (
    'warp_sum_f32', 'q8_top1_float_ordered_key', 'q8_top1_pack_key')) + '\n' +
    header[:header.index('static int mtp_native_refine_pair_enabled')])
binary = build / 'test'
subprocess.run(['nvcc', '-O3', '--use_fast_math', '-arch=sm_89',
    '-I' + str(build), str(root / 'ds4/tests/test_mtp_native_refine_pair.cu'),
    '-o', str(binary)], check=True)
subprocess.run([str(binary)], check=True)
