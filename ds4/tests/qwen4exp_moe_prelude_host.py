#!/usr/bin/env python3
"""Execute actual CUDA preparation bodies with host-simulated warp barriers.

This is an address/scheduling/arithmetic-order test, not CUDA numerical or
performance certification. Native device parity is in test_qwen4exp_moe.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=root / 'ds4_cuda_qwen4exp.cu')
parser.add_argument('--sanitize', action='store_true')
args = parser.parse_args()
source = args.source.read_text()

def extract(name, template=''):
    # These definitions close at column zero. Fail if a rename drops a body.
    at = source.index(name + '(')
    begin = source.rfind('\n', 0, at) + 1
    end = source.index('\n}', at) + 2
    result = source[begin:end].replace('extern __shared__', 'extern')
    assert '__device__' in result or '__global__' in result, name
    return template + result + '\n\n'

functions = [
    ('dev_qwen4exp_quantize_group', ''),
    ('qwen4exp_quantize_rows_kernel', ''),
    ('dev_qwen4exp_block_sum', ''),
    ('qwen4exp_router_select_topk_warp', 'template<bool Native>\n'),
    ('qwen4exp_router_select_topk_kernel', 'template<bool Native>\n'),
    ('qwen4exp_moe_group_small_block', ''),
    ('qwen4exp_moe_group_small_kernel', ''),
    ('qwen4exp_shared_gate_kernel', 'template<int RouterType = -1>\n'),
    ('qwen4exp_shared_gate_f32_block', ''),
    ('qwen4exp_moe_prelude_kernel', 'template<bool Native>\n'),
]
with tempfile.TemporaryDirectory(prefix='qwen-moe-prelude-') as tmp:
    tmp = Path(tmp)
    (tmp / 'moe_prelude_bodies.inc').write_text(''.join(extract(*f) for f in functions))
    at = source.index('extern "C" int ds4_gpu_qwen4exp_moe_prelude_tensor(')
    end = source.index('\n}', at) + 2
    (tmp / 'moe_prelude_api.inc').write_text(source[at:end])
    flags = ['-O2', '-std=c++17', '-march=native', '-ffp-contract=fast',
             '-Wall', '-Wextra', '-Werror', '-Wno-unknown-pragmas']
    if args.sanitize:
        # ucontext stacks are not supported reliably by ASan; use UBSan here.
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
    subprocess.run([os.environ.get('CXX', 'c++'), *flags, '-I', str(tmp),
                    str(root / 'tests/test_qwen4exp_moe_prelude_host.cpp'),
                    '-o', str(tmp / 'test')], check=True)
    subprocess.run([str(tmp / 'test')], check=True)
