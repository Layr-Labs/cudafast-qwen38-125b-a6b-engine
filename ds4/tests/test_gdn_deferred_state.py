#!/usr/bin/env python3
"""Experimental two-bank GDN transition log; not wired into production dispatch.

Extract the actual recurrence, preserve its arithmetic, and compare virtual final
and row-zero states against the existing eager final-state implementation.
"""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--arch', default='sm_89')
p.add_argument('--nvcc', default='nvcc')
p.add_argument('--compile-only', action='store_true')
p.add_argument('--benchmark', action='store_true')
p.add_argument('--production', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
src = (root / 'ds4/ds4_cuda_qwen4exp.cu').read_text()
out = root / '.build/gdn-deferred' / a.arch
out.mkdir(parents=True, exist_ok=True)

def body(name):
    pos = src.index(name + '(')
    start = src.rfind('\n', 0, pos) + 1
    end = src.index('\n}', pos) + 2
    return src[start:end]

base = body('qwen4exp_gdn_replay_gates_kernel')
candidate = 'template<unsigned Flush, bool Hybrid=false, bool FoldRow0=false>\n' + base.replace(
    'qwen4exp_gdn_replay_gates_kernel', 'deferred_kernel')

def change(old, new):
    global candidate
    assert candidate.count(old) == 1, old
    candidate = candidate.replace(old, new)

change('const uint32_t prefix = control ? *control : replay_rows;', '''
    const uint32_t word = *control;
    const uint32_t prefix = word & 0xffffu;
    const uint32_t bank = (word >> 16u) & 1u;
    const bool eager = Hybrid && (word & (1u << 17u));
    const bool flush = prefix >= Flush || (eager && prefix != 0u);
    const bool fold = FoldRow0 && flush;''')
change('if (prefix > DS4_QWEN4EXP_GDN_REPLAY_ROWS) return;',
       'if (prefix > Flush + 1u || bank > 1u || n_tokens != 2u) return;')
change('const uint32_t repeats = n_value_head / n_key_head;', '''
    float *const read_log = tape + (uint64_t)bank * (Flush + 2u) * tape_stride;
    float *const write_log = tape + (uint64_t)(bank ^ (uint32_t)flush) * (Flush + 2u) * tape_stride;
    const uint32_t repeats = n_value_head / n_key_head;''')
change('const float *const saved = tape +', 'const float *const saved = read_log +')
change('if (token == 0u && prefix < DS4_QWEN4EXP_GDN_REPLAY_ROWS) {',
       'if (token < 2u && (!fold || token != 0u)) {')
change('float *const record = tape + (uint64_t)prefix * tape_stride;',
       'float *const record = write_log + (uint64_t)((flush ? 0u : prefix) + token - (uint32_t)fold) * tape_stride;')
change('''            if (token == 0u && prefix == DS4_QWEN4EXP_GDN_REPLAY_ROWS)
                *(float4 *)(checkpoint + state_off) = h;''', '')
change('''        if (!replay) {
            const float4 q4''', '''        // Folding row zero makes the rollback snapshot itself the checkpoint.
        // The opposite bank retains row one; other CTAs may still read the old bank.
        if (flush && (fold ? (!replay && token == 0u)
                          : (replay && step + 1u == prefix)))
            *(float4 *)(checkpoint + state_off) = h;
        if (!replay) {
            const float4 q4''')
change('    *(float4 *)(state + state_off) = h;',
       '    if (eager) *(float4 *)(state + state_off) = h;')

# A pure materialization oracle: the original replay arithmetic with a wider
# validated replay-only bound, always called with n_tokens=0 and distinct output.
materialize = base.replace('qwen4exp_gdn_replay_gates_kernel', 'materialize_kernel')
materialize = materialize.replace('if (prefix > DS4_QWEN4EXP_GDN_REPLAY_ROWS) return;',
                                  'if (prefix > 10u || n_tokens != 0u) return;')
header = '''#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#define QWEN4EXP_GDN_DIM 128u
#define DS4_QWEN4EXP_GDN_REPLAY_ROWS 2u
'''
production = ''
if a.production:
    production = 'template<bool Precomputed>\n' + body('qwen4exp_gdn_deferred_kernel')
    production += r"""
#define DS4_QWEN4EXP_GDN_HEADS_TILED 1u
struct ds4_gpu_tensor { void *ptr; uint64_t bytes; int device=0; };
static bool glm53_cuda_tensor_has(const ds4_gpu_tensor*t,uint64_t n,uint64_t size) {
    return t && t->ptr && n <= t->bytes / size;
}
static int ds4_tensor_device_idx(const ds4_gpu_tensor*t){return t->device;}
static cudaStream_t cuda_decode_stream(){return nullptr;}
static int cuda_ok(cudaError_t e,const char*){return e==cudaSuccess;}
"""
    production += body('qwen4exp_replay_gate_disjoint')
    production += body('qwen4exp_deferred_range_safe')
    production += body('ds4_gpu_qwen4exp_gdn_deferred_materialize')
(out / 'kernels.cuh').write_text(header + '\n'.join([
    body('warp_sum_all_f32'), body('dot4_f32'), body('qwen4exp_gdn_sigmoid'),
    body('qwen4exp_gdn_softplus'), base, materialize, candidate, production]))
cmd = [a.nvcc, '-O3', '-std=c++17', '-arch=' + a.arch, '-ftz=false',
       '-prec-div=true', '-prec-sqrt=true', '-Xptxas=-v', '-I' + str(out),
       str(root / 'ds4/tests/test_gdn_deferred_state.cu')]
if a.production:
    cmd += ['-DGDN_DEFERRED_PRODUCTION=1']
binary = out / ('test.o' if a.compile_only else 'test')
if a.compile_only:
    cmd += ['-c']
cmd += ['-o', str(binary)]
with (out / 'compile.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Compiled', binary, flush=True)
if not a.compile_only:
    subprocess.run([str(binary)] + (['--benchmark'] if a.benchmark else []), check=True)
