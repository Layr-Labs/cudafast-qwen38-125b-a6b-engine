#!/usr/bin/env python3
"""Exercise the actual HC norm/quant kernels without loading model weights.

By default compare staged production code with its rolled implementation.
--baseline optionally supplies an earlier ds4_cuda_qwen4exp.cu for a third
comparison and CUDA-graph timing. PDL is disabled in this isolated test.
"""
import argparse
from pathlib import Path
import re
import subprocess


def extract(source, name, template=False):
    match = re.search(r"^__device__.*\b" + name + r"\(|^__global__.*\b" + name + r"\(", source, re.M)
    if not match:
        raise ValueError(f"missing production function: {name}")
    start = source.rfind("template <", 0, match.start()) if template else match.start()
    body = source.index("{", match.end())
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def kernel_source(source, namespace):
    names = ["warp_sum_all_f32", "qwen4exp_round_bf16", "qwen4exp_block_sum_f32",
             "qwen4exp_hc_norm_scale", "qwen4exp_hc_norm_scale_staged",
             "qwen4exp_hc_normed_value", "qwen4exp_q8_ftz", "qwen4exp_q8_rcp_approx"]
    if "qwen4exp_hc_block_sum_staged(" in source:
        names.insert(names.index("qwen4exp_hc_norm_scale_staged"), "qwen4exp_hc_block_sum_staged")
    pieces = [extract(source, name) for name in names]
    pieces.append(extract(source, "qwen4exp_hc_norm_quant_kernel", template=True))
    return "namespace " + namespace + " {\n" + "\n".join(pieces) + "\n}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / ".build" / "hc-norm-reuse"
    output.mkdir(parents=True, exist_ok=True)
    current = (root / "ds4/ds4_cuda_qwen4exp.cu").read_text()
    baseline = args.baseline.read_text() if args.baseline else current
    prefix = """#include <cuda_runtime.h>
#include <cstdint>
#define QWEN4EXP_HC_THREADS 256u
#define QWEN4EXP_HC_STAGED_STEPS 10u
#define QWEN4EXP_Q8_RCP127 0x1.020408p-7f
#define QWEN4EXP_PDL_TRIGGER() ((void)0)
#define QWEN4EXP_PDL_SYNC() ((void)0)
"""
    code = prefix + kernel_source(baseline, "baseline") + kernel_source(current, "candidate")
    (output / "kernels.cuh").write_text(code)
    exe = output / "test"
    subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=" + args.arch,
                    "-ftz=false", "-prec-div=true", "-prec-sqrt=true",
                    "-DBASELINE_STAGED=" + str(int(args.baseline is not None)),
                    "-I" + str(output), str(Path(__file__).with_suffix(".cu")),
                    "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
