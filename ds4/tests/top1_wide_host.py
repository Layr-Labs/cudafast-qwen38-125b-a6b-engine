#!/usr/bin/env python3
"""Host execution of actual CUDA top-1 bodies; no GPU performance claim."""
import argparse,os,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=root/'ds4_cuda.cu');p.add_argument('--sanitize',action='store_true');a=p.parse_args()
src=a.source.read_text()
code='struct indexer_top1_pair { float value; uint32_t index; };\n'
for name in ['topk_score_better','indexer_top1_kernel','indexer_top1_warp_reduce','indexer_top1_block_reduce','indexer_top1_chunks_kernel','indexer_top1_finish_kernel','indexer_top1_wide_shape','indexer_top1_ranges_overlap']:
 at=src.index(name+'(');start=src.rfind('\n',0,at)+1;end=src.index('\n}',at)+2;code+=src[start:end]+'\n'
with tempfile.TemporaryDirectory(prefix='top1-wide-') as d:
 d=Path(d);(d/'top1_bodies.inc').write_text(code)
 flags=['-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-unknown-pragmas','-Wno-misleading-indentation']
 if a.sanitize:flags+=['-fsanitize=undefined','-fno-sanitize-recover=all']
 exe=d/'test';subprocess.run([os.environ.get('CXX','c++'),*flags,'-I',str(d),str(root/'tests/test_top1_wide_host.cpp'),'-o',str(exe)],check=True);subprocess.run([str(exe)],check=True)
