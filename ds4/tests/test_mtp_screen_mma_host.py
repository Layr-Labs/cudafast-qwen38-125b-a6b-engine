#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
tests=Path(__file__).resolve().parent
s=(args.source or tests.parent/'ds4_cuda_mtp_screen_mma.cuh').read_text();s=s[s.index('__device__ __forceinline__ static uint32_t mtp_screen_weight_word'):]
shared='__shared__ __align__(16) unsigned char staged[16u * 816u];'
assert s.count(shared)==1
s=s.replace(shared,'unsigned char *staged = host_shared_storage + 16u;')
entry='const unsigned char *payload, unsigned byte) {'
assert s.count(entry)==1
s=s.replace(entry,entry+'\n    host_fragment_bounds(payload, byte);')
with tempfile.TemporaryDirectory(prefix='mtp-screen-mma-') as d:
 d=Path(d);(d/'mtp_screen_mma_bodies.inc').write_text(s)
 subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(d),str(tests/'test_mtp_screen_mma_host.cpp'),'-o',str(d/'test')],check=True)
 subprocess.run([str(d/'test')],check=True)
