#!/usr/bin/env python3
"""Extract the actual production kernel; emulate CUDA warp instructions only."""
import argparse,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
tests=Path(__file__).resolve().parent
source=(args.source or tests.parent/'ds4_cuda_hc_up_exact.cuh').read_text()
a=source.index('__device__ __forceinline__ static float hc_up_tree(')
b=source.index('static int hc_up_tuned_device')
body=source[a:b]
with tempfile.TemporaryDirectory(prefix='hc-up-exact-') as d:
 d=Path(d);(d/'hc_up_exact_bodies.inc').write_text(body)
 subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(d),str(tests/'test_hc_up_exact_mma_host.cpp'),'-o',str(d/'check')],check=True)
 subprocess.run([str(d/'check')],check=True)
