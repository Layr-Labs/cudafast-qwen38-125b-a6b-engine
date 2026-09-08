#!/usr/bin/env bash
#
# Thin proxy to the benchmark facade, which now lives at tools/benchmark.sh.
#
# This file's ONLY job is to (1) resolve the pinned benchd measurement binary,
# then (2) exec tools/benchmark.sh forwarding all args. It carries NO benchmark
# logic of its own -- the full benchmark lives in tools/benchmark.sh. Do NOT grow
# this file back into that; if you find yourself adding flags or mode handling
# here, they belong in the facade instead.
#
# WHAT CHANGED (benchd submodule -> channel-resolved prebuilt). benchd used to
# be a SHA-pinned SOURCE submodule at benchd/, and this proxy's steps 1-2
# installed that gitlink and compared the checkout against it. The submodule is
# gone. The benchmarker now ships as a prebuilt `benchd` binary published on the
# branch's dist channel and resolved by ./tools/fetch-benchd.sh, which verifies
# the bytes against the channel's own benchd.manifest.json before anything is
# installed. Two reasons: (a) the ranked M5 box has no Rust toolchain and could
# never build a source submodule, and (b) a participant who compiles the scorer
# can weaken it first, whereas a channel-published digest makes drift evident. The facade that lived at
# benchd/scripts/benchmark.sh was relocated to tools/benchmark.sh in the same
# change.
#
# Why this file must keep existing (the coverage boundary it restores):
# Gemma4Runtime.harnessHash() hashes a FIXED 9-root set that names "benchmark.sh"
# at index 4 (Sources/MLXFastTrustedHarness/Gemma4RuntimePreflight.swift and the
# identical Sources/MLXFastHarness/ copy). It resolves each root CWD-relative.
# With this file deleted -- as commit 92bdeccc's over-strip did -- only 8/9 roots
# hash and the harness hash is quietly dishonest. Restoring it makes the hash 9/9
# again. Its restoration and thinness are pinned by
# Tests/MLXFastTests/HarnessHashRootSetTests.swift; deleting or fattening this
# file turns those red.
#
# The benchmark.json manifest routes Yukon's ranked commands at
# ./tools/qwen38-125b-a6b-measure-and-score.sh (preSubmitCommand / benchmarkCommand), so
# this proxy is the developer/CWD-agnostic entry point and the harnessHash root,
# not a Yukon dependency.
set -euo pipefail

# Resolve the workspace root as the directory THIS script lives in, so the
# proxy behaves identically regardless of the caller's CWD. Step 3 below then
# makes this the PROCESS CWD, which is the invariant harnessHash() depends on.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHD_ENTRY="${SCRIPT_DIR}/tools/benchmark.sh"

if [[ ! -f "${BENCHD_ENTRY}" ]]; then
    echo "benchmark.sh: the benchmark facade is missing" >&2
    echo "  expected at: ${BENCHD_ENTRY}" >&2
    exit 1
fi

# 1. Resolve the PINNED measurement binary before anything else. fetch-benchd.sh
#    accepts an already-present benchd-bin/benchd whose sha256 and byte count
#    match the channel manifest beside it (the offline path the ranked box uses, where the binary
#    is scp'd in), otherwise downloads and verifies it. It NEVER yields an
#    unverified binary: a mismatch or a failed download is a hard refusal here,
#    exactly as an off-pin submodule checkout used to be.
#
#    This replaces the old gitlink check. The property is strictly stronger: the
#    submodule comparison bound the checked-out COMMIT and said nothing about the
#    working-tree bytes, so a benchd sitting at the pinned commit with a locally
#    edited script still passed. The sha256 below binds the actual bytes that run.
#
#    An explicit BENCHD= from the caller is passed through untouched -- that is
#    the caller deliberately naming another binary (benchd development, bisecting
#    a harness change), and it is not hash-checked.
if [[ -z "${BENCHD:-}" ]]; then
    BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
    export BENCHD
fi

# 2. Establish the workspace root as the PROCESS CWD before handing off. The
#    facade never cd's, and Gemma4Runtime.harnessHash() resolves its FIXED 9-root
#    set relative to the PROCESS CWD, so the Swift benchmark process the facade
#    launches must inherit CWD == this repo root or the 9-root hash silently
#    collapses toward the empty-set digest (the collapse is observable in
#    HarnessHashRootSetTests.swift Test C). Under `set -e` a failed cd aborts the
#    proxy rather than letting it run from the wrong root.
cd "${SCRIPT_DIR}"

# 3. Hand off to the facade, forwarding every argument unchanged.
exec "${BENCHD_ENTRY}" "$@"
