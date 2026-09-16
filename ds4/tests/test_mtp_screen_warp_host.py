from pathlib import Path
import subprocess,tempfile,argparse
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
s=(args.source or root.parent/'ds4_cuda_mtp_screen_warp.cuh').read_text();s=s[s.index('__global__'):]
a='__shared__ __align__(16) unsigned char staged[4u * 816u];';assert s.count(a)==1
s=s.replace(a,'unsigned char *staged = host_shared_storage + 16u;')
a='const unsigned char *payload = block + 2u;';assert s.count(a)==1
s=s.replace(a,a+' host_fragment_bounds(payload,0u);')
with tempfile.TemporaryDirectory(prefix='mtp-staged-warp-') as d:
 d=Path(d);(d/'bodies.inc').write_text(s)
 subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(d),str(root/'test_mtp_screen_warp_host.cpp'),'-o',str(d/'test')],check=True)
 subprocess.run([str(d/'test')],check=True)
