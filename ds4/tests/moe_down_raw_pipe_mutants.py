#!/usr/bin/env python3
"""Require real runtime failures from actual-source raw-pipeline corruptions."""
from pathlib import Path
import subprocess
import sys
import tempfile
root=Path(__file__).resolve().parents[1]
src=(root/'ds4_cuda_down_raw_pipe.cuh').read_text()
mutations={
 'wrong_expert_row': ('(uint64_t)(row0 + r) * down_row_bytes +','(uint64_t)(row0 + ((r + 1u) % 64u)) * down_row_bytes +'),
 'wrong_chunk_stride': ('(uint64_t)kc * 24u + piece * 16u','(uint64_t)kc * 20u + piece * 16u'),
 'wrong_prefetch_chunk': ('issue_raw(kc + QW_MMA_G);','issue_raw(kc);'),
 'omitted_last_piece': ('i < QW_DOWN_MMA_BM * 6u;','i + 1u < QW_DOWN_MMA_BM * 6u;'),
 'missing_wait': ('if (raw_pipe) qw_cpasync_wait0();','if (raw_pipe) {}'),
 'missing_commit': ('qw_cpasync_commit();','/* deliberately omitted */'),
 'missing_read_retirement': ('__syncthreads();\n            if (raw_pipe && kc + QW_MMA_G < groups)','/* deliberately omitted */\n            if (raw_pipe && kc + QW_MMA_G < groups)'),
 'wrong_shared_row': ('row * 24u + group * 6u','row * 20u + group * 6u'),
 'wrong_group_word': ('row * 24u + group * 6u','row * 24u + (group ^ 1u) * 6u'),
}
with tempfile.TemporaryDirectory(prefix='down-raw-pipe-mutants-') as tmp:
 for name,(a,b) in mutations.items():
  assert src.count(a)==1,(name,src.count(a))
  p=Path(tmp)/(name+'.cuh');p.write_text(src.replace(a,b))
  r=subprocess.run([sys.executable,str(root/'tests/moe_down_raw_pipe_host.py'),
      '--header',str(p),'--sanitize'],text=True,capture_output=True)
  out=r.stdout+r.stderr
  if r.returncode==0 or not ('FAIL:' in out or 'runtime error:' in out):
   print(out);raise SystemExit('mutant not rejected at runtime: '+name)
  line=next(x for x in out.splitlines() if 'FAIL:' in x or 'runtime error:' in x)
  print(name+': '+line,flush=True)
print(str(len(mutations))+' actual-source mutants caught at runtime PASS')
