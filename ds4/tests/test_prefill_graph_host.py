#!/usr/bin/env python3
"""Execute the actual graph controller against a deferred device-command model.

This checks captured identities/host transitions, not native CUDA execution.
--source permits runtime negative controls against the same test driver.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
s = (args.source or root/'ds4_qwen4exp_graph.inc').read_text()
actual = s[s.index('#ifdef DS4_TEST_HOOKS\n/* A replayed chunk'):
           s.index('\n/* Diagnostic/fallback bridge')]
support = (root/'tests/prefill_graph_host_support.cpp').read_text()
assert support.count('/* ACTUAL_CONTROLLER */') == 1
with tempfile.TemporaryDirectory(prefix='prefill-graph-') as tmp:
    tmp = Path(tmp)
    src, exe = tmp/'test.cpp', tmp/'test'
    src.write_text(support.replace('/* ACTUAL_CONTROLLER */', actual))
    subprocess.run(['c++', '-std=c++17', '-O1', '-g', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
