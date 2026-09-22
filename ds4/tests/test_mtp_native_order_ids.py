"""Compile and exercise the actual production select kernel without model weights.

Requires nvcc and a cooperative-launch CUDA device. This is a synthetic kernel
check, not a model-correctness test or an official performance measurement.
Generated source and binaries live outside editablePaths, under .build/.
"""
from pathlib import Path
import argparse
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--arch', default='sm_89')
p.add_argument('--nvcc', default='nvcc')
args = p.parse_args()
repo = Path(__file__).resolve().parents[2]
source = (repo / 'ds4/ds4_cuda_mtp_native.cuh').read_text()
start = source.index('#define MTP_NATIVE_SELECT_THREADS 256')
end = source.index('/* Zero restores the separate CUB ID sort.', start)
out = repo / '.build/native-order-test'
out.mkdir(parents=True, exist_ok=True)
(out / 'production_select.cuh').write_text(source[start:end])
binary = out / 'test_native_order'
subprocess.run([args.nvcc, '-O3', '--use_fast_math', '-std=c++17',
                '-arch=' + args.arch, '-I' + str(out),
                str(repo / 'ds4/tests/test_mtp_native_order_ids.cu'),
                '-o', str(binary)], check=True)
subprocess.run([str(binary)], check=True)
