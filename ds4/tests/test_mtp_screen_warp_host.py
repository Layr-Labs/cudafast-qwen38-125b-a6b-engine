from pathlib import Path
import subprocess,tempfile,argparse
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
tests=Path(__file__).resolve().parent
s=(args.source or tests.parent/'ds4_cuda_mtp_screen_warp.cuh').read_text();s=s[s.index('__global__'):]
entry='const unsigned char *payload = block + 2u;'
assert s.count(entry)==1
s=s.replace(entry,'host_warp_bounds(block); '+entry)
with tempfile.TemporaryDirectory(prefix='mtp-screen-warp-') as d:
 d=Path(d);(d/'mtp_screen_warp_bodies.inc').write_text(s)
 subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(d),str(tests/'test_mtp_screen_warp_host.cpp'),'-o',str(d/'test')],check=True)
 subprocess.run([str(d/'test')],check=True)
