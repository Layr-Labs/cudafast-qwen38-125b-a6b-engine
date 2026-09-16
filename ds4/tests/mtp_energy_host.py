#!/usr/bin/env python3
"""Run production CUDA screen bodies with host lane/barrier simulation.

This checks indexing, synchronization and the selection policy. It cannot
certify device rounding, CUDA execution or model acceptance/performance.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=root / 'ds4_cuda_mtp_native.cuh')
parser.add_argument('--sanitize', action='store_true')
args = parser.parse_args()
source = args.source.read_text()
cuda = (root / 'ds4_cuda.cu').read_text()

def definition(name):
    at = cuda.index(name + '(')
    begin = cuda.rfind('\n', 0, at) + 1
    end = cuda.index('\n}', at) + 2
    return cuda[begin:end] + '\n'

code = ''.join(definition(name) for name in (
    'warp_sum_f32', 'q8_top1_float_ordered_key', 'q8_top1_pack_key'))
code += source[:source.index('struct mtp_native_layout')]
with tempfile.TemporaryDirectory(prefix='mtp-energy-host-') as tmp:
    tmp = Path(tmp)
    (tmp / 'mtp_energy_bodies.inc').write_text(code)
    flags = ['-O2', '-std=c++17', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas']
    if args.sanitize:
        # ASan does not reliably support these ucontext stacks.
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                    str(root / 'tests/test_mtp_energy_host.cpp'),
                    '-o', str(tmp / 'test')], check=True)
    subprocess.run([str(tmp / 'test')], check=True)
