#!/usr/bin/env bash
#
# three-flows-dry-run.sh -- run the track's THREE FLOWS off the box, against the
# resident STUB engine, and record what each step did.
#
# WHAT IT IS FOR. The ranked box is scarce and a ranked window costs a 103.7 GiB
# load. Every plumbing break that a laptop can find must be found on a laptop
# first. This script runs, on a host with NO GPU, NO CUDA and NO checkpoint:
#
#   FLOW A  calibration / the paired official path
#           tools/qwen38-125b-a6b-measure-and-score.sh --preflight-only on the
#           stock tree, then the FULL measure-and-score on an ARMED scratch
#           worktree, at the serial declaration and at the mtp1 declaration.
#   FLOW B  the self-benchmark (local iterate)
#           benchd iterate --mode local-iterate at depth 0 and at --mtp-depth 1.
#   FLOW C  the server benchmark pipeline (the ranked job's own steps)
#           tools/ranked-box-preflight.sh, then the serve-up resident boot, then
#           measure-and-score, then score.json.
#
# THE ENGINE IS THE STUB. tools/ds4/resident-stub-engine.c is a synthetic
# ds4_shim.h engine. The REAL tools/serve-up.sh boots the REAL
# harness/protocol-adapter/ds4_shim/ds4_resident.c over it, and the REAL
# `cuda-engine` adapter connects. So the LIFECYCLE and the WIRE are the real
# ones; the TOKENS and every timing are meaningless. Correctness fails by
# construction here, and a failed correctness gate is a RESULT of this run, not
# a fault in it: what this script reports is the exit code of each step and the
# fields each step sealed.
#
# WHAT IT REPORTS. A PASS/FAIL TABLE, keyed on the exit code the design says
# each step must produce. Nearly every gate on this track is shut today -- the
# fixture is unarmed, the staged goldens name the retired reference model, and
# the track's official baseline is PENDING-ORGANIZER -- so nearly every step
# refuses, and a table of bare exit codes cannot tell a correct refusal from a
# plumbing break. Each step therefore declares the code it must give AND, for a
# refusal, a string that refusal must print. An expected refusal is a PASS. A
# step that refuses for another reason, or stops refusing, is a FAIL. The sweep
# exits non-zero only on such an unexpected outcome.
#
# THE SEALED FIELDS ARE CHECKED, NOT ONLY PRINTED. A leg whose step passed must
# have sealed its score.json, and the checks read that file: the spec the engine
# echoed (effective_spec_mode/effective_spec_depth), the speculative counters
# (spec_rounds, spec_drafted_total, spec_accepted_total, and
# spec_verify_replay_disagreements when the engine reports it), and the backend
# identity. While the official baseline is PENDING-ORGANIZER benchd seals NO
# artifact, so those checks have nothing to read and say so by name.
#
# BOX-ONLY ASSERTIONS ARE NAMED, NEVER SKIPPED SILENTLY. Three things cannot be
# checked off the box, and each is printed as a SKIPPED-BOX-ONLY line that names
# the assertion and what was substituted for it:
#   1. ranked-box-preflight.sh section 2b/2c, the GPU temperature reader
#      (macmon on macOS, nvidia-smi on Linux).
#   2. ranked-box-preflight.sh section 7, the nvcc / driver toolchain pins.
#   3. ./setup.sh, which builds the CUDA engine and verifies the 111 GB GGUF
#      snapshot. It is not run at all.
# Nothing else is relaxed. The staged-asset pins, the arm gate, the golden
# integrity pin, the engine-pin/weight-owner/depth pins and the whole benchd
# scored path run exactly as they do on the box.
#
# Usage:
#   tools/three-flows-dry-run.sh
#
# Env:
#   THREE_FLOWS_WORK   work directory (default: a fresh mktemp -d, kept)
#   BENCHD             benchd binary. Default: tools/fetch-benchd.sh, which
#                      needs the dist pair already staged in BENCHD_BIN_DIR or a
#                      dist token.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
# The synthetic artifact and the synthetic engine declaration. One writer, so
# this sweep and the tools tests cannot drift apart.
# shellcheck source=tools/ds4/synthetic-window.sh
. "${ROOT_DIR}/tools/ds4/synthetic-window.sh"
WORK="${THREE_FLOWS_WORK:-$(mktemp -d)}"
mkdir -p "${WORK}"

say()  { printf '\n=== %s\n' "$*"; }
note() { printf '     %s\n' "$*"; }
# An assertion this host cannot make, with what stands in for it.
skip() { printf '     SKIPPED-BOX-ONLY: %s\n' "$*"; }
# A step this host did not run at all. NOT box-only, and nothing stands in for
# it: the sweep is simply missing that coverage and says so.
notrun() { printf '     NOT RUN HERE: %s\n' "$*"; }
die()  { printf 'three-flows-dry-run: %s\n' "$*" >&2; exit 1; }

STEPS_TSV="${WORK}/steps.tsv"
LEGS_TSV="${WORK}/legs.tsv"
: > "${STEPS_TSV}"
: > "${LEGS_TSV}"

# EVERY STEP DECLARES THE EXIT CODE IT MUST PRODUCE, and a refusal that is the
# right answer is a PASS. Most of this sweep runs against gates that are SHUT
# today -- an unarmed fixture, goldens under the retired reference model, a
# track whose official baseline the organizer has not captured -- so a table of
# bare exit codes says nothing: nearly every line is a 1, and a plumbing break
# looks exactly like a correct refusal.
#
# So a step carries three things:
#   EXPECT  the exit code the design says it must produce
#   NEEDLE  when EXPECT is not 0, a string the refusal must print. The step
#           passes only when the run refuses FOR THAT REASON, so a break that
#           happens to exit 1 is still a FAIL.
#   WHY     one line naming the gate, printed under the table.
#
# A step that stops refusing is a FAIL as well, and the message says the
# expectation is stale. That is deliberate: the day a gate opens, this sweep
# must be re-read rather than kept green.
#
#   step NAME EXPECT NEEDLE WHY -- COMMAND...
step() {
  local name="$1" expect="$2" needle="$3" why="$4"; shift 4
  [[ "${1:-}" == "--" ]] || die "step ${name}: the command must follow --"
  shift
  local log rc=0 verdict
  log="${WORK}/$(printf '%s' "${name}" | tr ' /' '__').log"
  say "${name}"
  "$@" > "${log}" 2>&1 || rc=$?
  note "exit ${rc}   want ${expect}   log: ${log}"
  tail -3 "${log}" | sed 's/^/     | /'

  if [[ "${rc}" != "${expect}" ]]; then
    if [[ "${expect}" != "0" && "${rc}" == "0" ]]; then
      verdict="FAIL-STALE"
      note "FAIL: this step no longer refuses. The expectation is stale: re-read the step, and arm what the refusal was holding back."
    else
      verdict="FAIL"
      note "FAIL: exit ${rc}, and the design says ${expect}."
    fi
  elif [[ "${expect}" == "0" ]]; then
    verdict="PASS"
  elif ! grep -qF -- "${needle}" "${log}"; then
    verdict="FAIL-OTHER"
    note "FAIL: it refused with exit ${rc} as expected, but not for the expected reason (no '${needle}' in the log)."
  else
    verdict="PASS-EXPECTED-REFUSAL"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${expect}" "${rc}" "${verdict}" "${why}" "${log}" >> "${STEPS_TSV}"
  return 0
}

# A measured leg and the artifact benchd seals for it. The sealed-field checks
# read THIS: a leg whose step passed with exit 0 must have sealed its
# score.json, and a leg the table shows refusing seals nothing.
#   leg STEP_NAME SCORE_PATH serial|<depth>
leg() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${LEGS_TSV}"
}

command -v jq >/dev/null 2>&1 || die "jq is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required to build the cuda-engine adapter"
command -v cc >/dev/null 2>&1 || die "cc is required to build the resident against the stub engine"

# --- the pinned benchd -----------------------------------------------------
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${ROOT_DIR}/tools/fetch-benchd.sh")" \
    || die "no benchd: stage the dist pair in BENCHD_BIN_DIR, or set BENCHD"
fi
[[ -x "${BENCHD}" ]] || die "benchd is not executable: ${BENCHD}"
export BENCHD

# --- build the resident over the stub engine, and the adapter ----------------
say "build: the real ds4_resident.c over tools/ds4/resident-stub-engine.c, and cuda-engine"
SHIM="${ROOT_DIR}/harness/protocol-adapter/ds4_shim"
cc -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Werror -I "${SHIM}" \
  -o "${WORK}/ds4-resident" "${SHIM}/ds4_resident.c" "${ROOT_DIR}/tools/ds4/resident-stub-engine.c" \
  || die "the resident does not build against the stub engine"
cargo build --quiet --manifest-path "${ROOT_DIR}/harness/protocol-adapter/Cargo.toml" --bin cuda-engine \
  || die "cuda-engine does not build"
ENGINE="${ROOT_DIR}/harness/protocol-adapter/target/debug/cuda-engine"
note "resident: ${WORK}/ds4-resident"
note "adapter:  ${ENGINE}"

# --- a synthetic window ------------------------------------------------------
# The window serve-up.sh plans BEFORE it boots anything: a 2-shard body with a
# REAL GGUF tensor index, a draft head, the pinned engine's memory declaration,
# and a meminfo the plan can read. This host has no /proc/meminfo, no ds4
# checkout and no 103.7 GiB artifact, and the plan reads all three, so all three
# are written here by the one writer the other off-box drivers use.
WEIGHTS="${WORK}/weights"
synthetic_window_weights "${WEIGHTS}" || die "cannot write the synthetic artifact"
# The declaration for the flows that run the STOCK tree's serve-up.sh. The real
# tree is never written to, so it is named through SERVE_UP_ENGINE_HEADER. The
# armed scratch worktree below carries its own at the DEFAULT path instead, so
# the default resolution is exercised too.
ENGINE_HEADER="${WORK}/ds4_qwen4exp.h"
synthetic_window_engine_header "${ENGINE_HEADER}"
printf 'MemTotal:       268435456 kB\nMemAvailable:   268435456 kB\n' > "${WORK}/meminfo"

# The track goldens are organizer material published in R2 and staged on the
# box out of band; no tree here carries them, so the staged directory is
# named and required.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
[ -n "${GOLDEN_DIR}" ] || die "MLXFAST_QWEN38_GOLDEN_DIR is unset; stage the track goldens and export the directory"
[ -d "${GOLDEN_DIR}" ] || die "MLXFAST_QWEN38_GOLDEN_DIR is not a directory: ${GOLDEN_DIR}"
CONTRACT="${ROOT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"

serve() {
  # Boot ONE stub resident for a window and run the rest against it, through the
  # real serve-up.sh.
  SERVE_UP_WEIGHTS_DIR="${WEIGHTS}" \
  SERVE_UP_LOG_DIR="${WORK}/serve" \
  SERVE_UP_RESIDENT_BIN="${WORK}/ds4-resident" \
  SERVE_UP_HEALTH_TIMEOUT_S=60 \
  SERVE_UP_MEMINFO="${WORK}/meminfo" \
  SERVE_UP_ENGINE_HEADER="${ENGINE_HEADER}" \
  "${ROOT_DIR}/tools/serve-up.sh" "$@"
}

# =============================================================================
# THE GATES THAT ARE SHUT TODAY
# =============================================================================
# Three of them, each closed for a stated reason, each the expected answer of
# several steps below. They are named ONCE here and referred to by name.
#
#   UNARMED     the track fixture declares official_scoring_enabled: false, so
#               the arm gate refuses every official path.
#   GOLDENS     the staged correctness goldens are authored under the RETIRED
#               NVFP4 reference model, and the fixture pins the unsloth GGUF
#               model, so validate-golden refuses them by provenance. They are
#               re-authored on the box (tools/qwen4exp-golden-reauthor.sh).
#   BASELINE    benchd resolves this track's official baseline as
#               QWEN38-125B-A6B-CUDA-PENDING-ORGANIZER and refuses to score
#               BEFORE the first timed phase. No leg reaches a timed window,
#               here or on the box, and no score.json is sealed anywhere, so
#               the sealed-field checks below have nothing to read yet. The
#               --baseline-* flags do not lift it: they override the pair, and
#               this refusal is about the track constants.
NEEDLE_UNARMED="does not declare official_scoring_enabled: true"
NEEDLE_GOLDENS="model_provenance does not match the pinned reference model"
NEEDLE_BASELINE="QWEN38-125B-A6B-CUDA-PENDING-ORGANIZER"
WHY_UNARMED="UNARMED: the stock fixture declares official_scoring_enabled: false, and the arm gate must refuse it"
WHY_GOLDENS="GOLDENS: the staged goldens name the retired NVFP4 reference model, and the fixture pins the unsloth GGUF model"
WHY_BASELINE="BASELINE: the track's official baseline is PENDING-ORGANIZER, so benchd refuses before the first timed phase"

# =============================================================================
# FLOW A -- calibration / the paired official path
# =============================================================================
step "flow A: measure-and-score --preflight-only on the stock tree (arm gate)" \
  1 "${NEEDLE_UNARMED}" "${WHY_UNARMED}" -- \
  env MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
      "${ROOT_DIR}/tools/qwen38-125b-a6b-measure-and-score.sh" --preflight-only

# The ARMED legs need a fixture that declares official_scoring_enabled: true.
# The contract path in measure-and-score.sh is fixed, so the arming is done in a
# SCRATCH WORKTREE and the real tree is never touched. The worktree keeps the
# git identity, so the ds4 gitlink the serve identity records is the real pin.
SHADOW="${WORK}/armed-tree"
if git -C "${ROOT_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "${ROOT_DIR}" worktree add --detach --quiet "${SHADOW}" HEAD
  trap 'git -C "${ROOT_DIR}" worktree remove --force "${SHADOW}" >/dev/null 2>&1 || true' EXIT
  # The worktree starts at HEAD, so it would run the COMMITTED scripts. Carry
  # the working tree's own tracked edits across, or the sweep reports the state
  # of HEAD while the author is reading it as the state of their branch.
  git -C "${ROOT_DIR}" diff HEAD > "${WORK}/worktree.patch"
  if [[ -s "${WORK}/worktree.patch" ]]; then
    git -C "${SHADOW}" apply "${WORK}/worktree.patch" \
      || die "the working tree's uncommitted diff does not apply to the scratch worktree"
    note "carried the working tree's uncommitted diff into the scratch worktree"
  fi
  python3 - "${SHADOW}/fixtures/qwen3_8_125b_a6b_track.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["official_scoring_enabled"] = True
open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
  # The pinned engine's memory declaration, at the path serve-up.sh resolves by
  # DEFAULT (<tree>/ds4/ds4_qwen4exp.h). The ds4 submodule is not checked out in
  # a scratch worktree, so without it the memory plan refuses every official leg
  # before the boot. Writing it HERE and not in the real tree keeps the default
  # resolution under test and leaves the author's checkout untouched.
  synthetic_window_engine_header "${SHADOW}/ds4/ds4_qwen4exp.h"
  note "armed scratch worktree: ${SHADOW} (official_scoring_enabled: true, synthetic ds4/ds4_qwen4exp.h)"

  run_official() {
    local score="$1"
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    MLXFAST_ENGINE_BIN="${ENGINE}" \
    MLXFAST_WEIGHTS_PATH="${WEIGHTS}" \
    MLXFAST_SCORE_PATH="${score}" \
    SERVE_UP_LOG_DIR="${WORK}/serve" \
    SERVE_UP_RESIDENT_BIN="${WORK}/ds4-resident" \
    SERVE_UP_MEMINFO="${WORK}/meminfo" \
    SERVE_UP_HEALTH_TIMEOUT_S=60 \
      "${SHADOW}/tools/qwen38-125b-a6b-measure-and-score.sh" "${@:2}"
  }

  step "flow A: measure-and-score --preflight-only on the ARMED tree (arm gate + golden pin)" \
    1 "${NEEDLE_GOLDENS}" "${WHY_GOLDENS}" -- \
    run_official "${WORK}/unused.json" --preflight-only

  step "flow A: measure-and-score FULL, serial declaration (serve-up + benchd official seal)" \
    1 "${NEEDLE_BASELINE}" "${WHY_BASELINE}" -- \
    run_official "${WORK}/score.official.serial.json"
  leg "flow A: measure-and-score FULL, serial declaration (serve-up + benchd official seal)" \
    "${WORK}/score.official.serial.json" serial

  python3 - "${SHADOW}/mtp-head.manifest.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["spec"] = {"enabled": True, "num_speculative_tokens": 1}
open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
  note "armed worktree declaration set to mtp1 (spec.enabled true, num_speculative_tokens 1)"

  step "flow A: measure-and-score FULL, mtp1 declaration (--mtp-depth 1 on the official argv)" \
    1 "${NEEDLE_BASELINE}" "${WHY_BASELINE}" -- \
    run_official "${WORK}/score.official.mtp1.json"
  leg "flow A: measure-and-score FULL, mtp1 declaration (--mtp-depth 1 on the official argv)" \
    "${WORK}/score.official.mtp1.json" 1
else
  notrun "the ARMED official legs and the armed ranked-box-preflight: ${ROOT_DIR} is not a
          git checkout, so no scratch worktree can be cut and the fixture cannot be armed
          without editing the real tree. Run this from a git checkout to cover them."
fi

# =============================================================================
# FLOW B -- the self-benchmark (local iterate), depth 0 and depth 1
# =============================================================================
TRACK_ID="$(jq -r .trackId "${ROOT_DIR}/benchmark.json")"

# A subshell, so the per-leg serve spec never leaks into the next flow.
run_local_iterate() (
  local depth="$1" golden="$2" score="$3"; shift 3
  export MLXFAST_QWEN_MTP_TRACK_ID="${TRACK_ID}"
  if [[ "${depth}" == "0" ]]; then
    export SERVE_UP_SPECULATIVE=0
  else
    export SERVE_UP_SPECULATIVE=1 SERVE_UP_SPEC_DRAFT_LEN="${depth}"
  fi
  serve "${BENCHD}" iterate \
    --engine "${ENGINE}" \
    --weights "${WEIGHTS}" \
    --golden "${golden}" \
    --mode local-iterate \
    --score-path "${score}" \
    "$@"
)

step "flow B: local-iterate depth 0 (serial)" \
  1 "${NEEDLE_BASELINE}" "${WHY_BASELINE}" -- \
  run_local_iterate 0 "${GOLDEN_DIR}/botany.golden.json" "${WORK}/score.local-iterate.json"
leg "flow B: local-iterate depth 0 (serial)" "${WORK}/score.local-iterate.json" serial

step "flow B: local-iterate --mtp-depth 1" \
  1 "${NEEDLE_BASELINE}" "${WHY_BASELINE}" -- \
  run_local_iterate 1 "${GOLDEN_DIR}/botany.mtp1.golden.json" "${WORK}/score.mtp1.json" \
    --mtp-depth 1
leg "flow B: local-iterate --mtp-depth 1" "${WORK}/score.mtp1.json" 1

# =============================================================================
# FLOW C -- the ranked job's steps
# =============================================================================
# Step: ranked-box-preflight.sh. Two of its assertions are box-only and each is
# named here with what stands in for it.
skip "ranked-box-preflight section 2b/2c, the GPU temperature reader (macmon / nvidia-smi): substituted MLXFAST_GPU_TEMP_CMD='echo 45', benchd's own documented reader override. THERMAL CONTROL IS NOT PROVEN BY THIS RUN."
skip "ranked-box-preflight section 7, the nvcc release and the nvidia-smi driver floor: substituted stub nvcc/nvidia-smi on PATH reporting the fixture's pinned values. THE TOOLCHAIN EPOCH IS NOT PROVEN BY THIS RUN."
skip "./setup.sh (the ds4 CUDA build and the 111 GB GGUF snapshot verification): not run at all."

STUBBIN="${WORK}/box-stubs"
mkdir -p "${STUBBIN}"
want_nvcc="$(jq -r '.serve_configuration.toolchain.nvcc_version' "${CONTRACT}")"
want_driver="$(jq -r '.serve_configuration.toolchain.driver_min' "${CONTRACT}")"
cat > "${STUBBIN}/nvcc" <<EOF
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 13.0, ${want_nvcc}"
EOF
cat > "${STUBBIN}/nvidia-smi" <<EOF
#!/usr/bin/env bash
echo "${want_driver}"
EOF
chmod +x "${STUBBIN}/nvcc" "${STUBBIN}/nvidia-smi"

preflight_in() {
  # $1 = the tree to run the preflight of. BENCHD is UNSET for it: section 2
  # refuses a caller-supplied benchd on a ranked box, and that refusal is
  # correct, so the pinned pair is named through BENCHD_BIN_DIR instead.
  env -u BENCHD \
      PATH="${STUBBIN}:${PATH}" \
      MLXFAST_GPU_TEMP_CMD="echo 45" \
      MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
      BENCHD_BIN_DIR="$(cd -- "$(dirname -- "${BENCHD}")" && pwd -P)" \
      "$1/tools/ranked-box-preflight.sh"
}

step "flow C: ranked-box-preflight.sh on the stock tree (two box-only assertions substituted, named above)" \
  1 "${NEEDLE_UNARMED}" "${WHY_UNARMED}" -- \
  preflight_in "${ROOT_DIR}"

# The stock tree is UNARMED, so the run above stops at section 5 and sections 6
# and 7 -- the benchd manifest, the ds4 engine pin, the weight owner and the
# depth pins -- are never reached. Run it again on the armed scratch tree so
# they are.
if [[ -d "${SHADOW}" ]]; then
  step "flow C: ranked-box-preflight.sh on the ARMED tree (reaches sections 6 and 7)" \
    0 "" "every section runs: the fixture is armed and the goldens are staged and pinned" -- \
    preflight_in "${SHADOW}"
fi

step "flow C: fetch-benchd.sh (the offline pair verify the ranked job runs)" \
  0 "" "the staged pair verifies against its own manifest, with no network" -- \
  env BENCHD_BIN_DIR="$(cd -- "$(dirname -- "${BENCHD}")" && pwd -P)" \
      "${ROOT_DIR}/tools/fetch-benchd.sh"

step "flow C: spec-declaration.sh describe / speculative / draft-len" \
  0 "" "the stock declaration is serial, and the three verbs agree on it" -- \
  bash -c '"$1"/tools/spec-declaration.sh describe && "$1"/tools/spec-declaration.sh speculative && "$1"/tools/spec-declaration.sh draft-len' _ "${ROOT_DIR}"

# =============================================================================
# WHAT WAS SEALED -- the sealed-field CHECKS
# =============================================================================
# The window identity first, then the artifacts. A leg whose step passed with
# exit 0 MUST have sealed its score.json, and this reads that file: the spec the
# engine echoed, the speculative counters, and the backend identity benchd
# sealed. A leg that the table shows refusing seals nothing. Such a leg is
# reported as a NAMED GAP with the gate that stopped it, never as a silent skip.
say "sealed fields"
# `set -e` is on and the block below exits 1 when a sealed field is wrong, so
# the exit code is captured HERE, on the command itself. A bare `SEALED_RC=$?`
# after the heredoc never ran: the shell had already left.
SEALED_RC=0
python3 - "${WORK}" "${STEPS_TSV}" "${LEGS_TSV}" <<'SEALEDPY' || SEALED_RC=$?
import json, os, sys

work, steps_tsv, legs_tsv = sys.argv[1], sys.argv[2], sys.argv[3]
failures = []
gaps = []


def read_tsv(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if line:
                    rows.append(line.split("\t"))
    return rows


steps = {r[0]: {"expect": r[1], "rc": r[2], "verdict": r[3], "why": r[4]}
         for r in read_tsv(steps_tsv)}

# The window. Its presence decides which backend identity a sealed artifact
# must carry: a resident-attached worker seals the resident's topology and its
# load epoch, and a worker with no resident seals the mock.
identity_path = os.path.join(work, "serve", "serve-identity.json")
identity = None
if os.path.exists(identity_path):
    identity = json.load(open(identity_path, encoding="utf-8"))
    print("  serve-identity.json:")
    for key in ("weight_owner", "engine_pin", "engine_ident", "spec_config",
                "declared_draft_len", "mtp_draft_tokens", "resident_pid"):
        print(f"      {key:<26} = {json.dumps(identity.get(key))[:100]}")
    hello = identity.get("hello") or {}
    for key in ("load_epoch", "mtp_armed", "draft_tokens", "ident"):
        print(f"      hello.{key:<20} = {json.dumps(hello.get(key))[:100]}")
else:
    print("  serve-identity.json: ABSENT (no window booted)")


def check(condition, message):
    if not condition:
        failures.append(message)
    return condition


def check_leg(label, path, depth):
    """Assert the fields benchd sealed for one leg."""
    if not os.path.exists(path):
        failures.append(f"{label}: the step passed, so {path} must be sealed, and it is absent")
        return
    try:
        doc = json.load(open(path, encoding="utf-8"))
    except Exception as err:                    # the sweep reports it, never raises
        failures.append(f"{label}: {path} is unreadable ({err})")
        return
    metrics = doc.get("metrics") or {}
    per_prompt = metrics.get("per_prompt")
    first = per_prompt[0] if isinstance(per_prompt, list) and per_prompt else {}
    print(f"  {os.path.basename(path)}: passed={doc.get('passed')} score={doc.get('score')}")
    for key in ("effective_spec_mode", "effective_spec_depth", "engine_backend",
                "engine_device", "engine_protocol_version"):
        shown = "= " + json.dumps(metrics.get(key))[:80] if key in metrics else "ABSENT"
        print(f"      {key:<32} {shown}")
    for key in ("spec_rounds", "spec_drafted_total", "spec_accepted_total",
                "spec_verify_replay_disagreements"):
        shown = "= " + json.dumps(first.get(key))[:80] if key in first else "ABSENT"
        print(f"      per_prompt[0].{key:<19} {shown}")

    # THE SPEC THE ENGINE ECHOED. benchd discards a leg whose echo disagrees
    # with the request, so a sealed artifact carries the spec that really ran.
    mode = metrics.get("effective_spec_mode")
    sealed_depth = metrics.get("effective_spec_depth")
    if depth == "serial":
        check(mode != "mtp", f"{label}: a serial leg sealed effective_spec_mode {mode!r}")
        check(sealed_depth in (0, None),
              f"{label}: a serial leg sealed effective_spec_depth {sealed_depth!r}")
    else:
        want = int(depth)
        check(mode == "mtp", f"{label}: effective_spec_mode is {mode!r}, and the leg requested mtp")
        check(sealed_depth == want,
              f"{label}: effective_spec_depth is {sealed_depth!r}, and the leg requested {want}")
        for key in ("spec_rounds", "spec_drafted_total", "spec_accepted_total"):
            check(key in first, f"{label}: per_prompt[0] carries no {key}")
        rounds, drafted = first.get("spec_rounds"), first.get("spec_drafted_total")
        if isinstance(rounds, int) and isinstance(drafted, int):
            check(rounds >= 1 and drafted >= rounds,
                  f"{label}: an mtp{want} leg drafted {drafted} tokens over {rounds} rounds")
    # Reported only when the engine reports it, and then it must be zero: a
    # disagreement means the verify replay did not reproduce the draft.
    if "spec_verify_replay_disagreements" in first:
        check(first["spec_verify_replay_disagreements"] == 0,
              f"{label}: spec_verify_replay_disagreements is "
              f"{first['spec_verify_replay_disagreements']!r}")

    # THE BACKEND IDENTITY. The resident-attached worker seals the topology and
    # the load epoch. A worker with no resident seals the mock.
    backend = metrics.get("engine_backend")
    device = metrics.get("engine_device")
    if identity is not None:
        check(isinstance(backend, str) and backend.startswith("ds4-resident load_epoch="),
              f"{label}: a window booted, and engine_backend is {backend!r}")
    else:
        check(backend == "mock" and device == "none",
              f"{label}: no window booted, and engine_backend/engine_device are "
              f"{backend!r}/{device!r}")


for step_name, path, depth in read_tsv(legs_tsv):
    label = os.path.basename(path)
    record = steps.get(step_name)
    if record is None:
        failures.append(f"{label}: no step is recorded for '{step_name}'")
        continue
    if record["rc"] != "0":
        gaps.append(f"{label}: no artifact is sealed -- {record['why']}")
        continue
    check_leg(label, path, depth)

for line in gaps:
    print(f"     NOT RUN HERE: {line}")
for line in failures:
    print(f"     SEALED-FIELD FAIL: {line}")
# The count of legs that sealed nothing, so the closing line can say how much of
# this sweep the shut gates held back.
with open(os.path.join(work, "sealed-gaps.count"), "w", encoding="utf-8") as handle:
    handle.write(f"{len(gaps)}\n")
raise SystemExit(1 if failures else 0)
SEALEDPY

# =============================================================================
# THE TABLE
# =============================================================================
say "steps: the verdict, the exit code, and the exit code the design wants"
printf '  %-22s %4s %4s  %s\n' "VERDICT" "exit" "want" "step"
awk -F'\t' '{printf "  %-22s %4s %4s  %s\n", $4, $3, $2, $1}' "${STEPS_TSV}"
if [[ "${SEALED_RC}" -eq 0 ]]; then SEALED_VERDICT="PASS"; else SEALED_VERDICT="FAIL"; fi
printf '  %-22s %4s %4s  %s\n' "${SEALED_VERDICT}" "${SEALED_RC}" "0" \
  "sealed-field checks over the artifacts the passing legs sealed"

say "why each refusal is the expected answer"
awk -F'\t' '$2 != "0" {printf "  %s\n      %s\n", $1, $5}' "${STEPS_TSV}"

say "work directory: ${WORK}"

UNEXPECTED="$(awk -F'\t' '$4 !~ /^PASS/ {c++} END {print c + 0}' "${STEPS_TSV}")"
GAPS="$(cat "${WORK}/sealed-gaps.count" 2>/dev/null || echo 0)"
if [[ "${UNEXPECTED}" -eq 0 && "${SEALED_RC}" -eq 0 ]]; then
  say "OK: every step gave the exit code the design wants"
  if [[ "${GAPS}" -gt 0 ]]; then
    note "${GAPS} leg(s) sealed no artifact, so their sealed fields are UNCHECKED here. Each one is named above with the gate that stopped it."
  fi
  exit 0
fi
say "FAILED: ${UNEXPECTED} step(s) did not give the expected outcome; the sealed-field checks exit ${SEALED_RC}"
exit 1
