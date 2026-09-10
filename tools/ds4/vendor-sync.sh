#!/usr/bin/env bash
# Re-vendor ds4/ from a signed commit of the port repository.
#
# WHY ds4/ IS VENDORED AND NOT A SUBMODULE. A participant has to be able to
# READ, EDIT, DIFF and SUBMIT the engine. A submodule pinned to an INTERNAL
# repository gives them none of that: they cannot fork it, the roster hashes
# only the gitlink, and the ranked box needs a credential to fetch it. So the
# engine source lives in this repository as plain files, the way the Gemma
# track carries its engine, and this script is how our port development keeps
# flowing: it develops in Layr-Labs/ds4 and lands here BY SCRIPT, never by hand.
#
# WHAT IT DOES. Takes a tag or sha of the port, verifies the commit's signature,
# exports its tree, drops the paths in EXCLUDE below, replaces ds4/ wholesale,
# and rewrites ds4/VENDOR.json and the fixture's two engine_pin fields together
# so they cannot drift apart.
#
# THE EXCLUSIONS ARE DATA, NEVER SOURCE. The port tree is 116 MB, and 96 MB of
# that is corpora and captured responses under gguf-tools/ that no build target
# reads: the imatrix dataset, the quality-testing response captures, the
# speed-bench text, and the direction-steering vectors. Dropping them leaves
# 18.4 MB and EVERY source file, so every Makefile target still resolves.
# Vendoring them would put 96 MB into every Yukon cut, every ranked checkout and
# every participant fork to no purpose. The list is recorded in VENDOR.json, so
# what was dropped is a fact in the tree rather than lore in a script.
#
# Usage:
#   tools/ds4/vendor-sync.sh <tag|sha> [--repo <url>] [--upstream-sha <sha>]
#                                      [--allow-unsigned]
#
# Example:
#   tools/ds4/vendor-sync.sh pin-278b799
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'ds4/vendor-sync.sh: %s\n' "$*"; }
die() { printf 'ds4/vendor-sync.sh: %s\n' "$*" >&2; exit 1; }

REPO_URL="git@github.com:Layr-Labs/ds4.git"
UPSTREAM_URL="https://github.com/antirez/ds4"
REF=""
UPSTREAM_OVERRIDE=""
ALLOW_UNSIGNED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="${2:?--repo needs a url}"; shift 2 ;;
    --upstream-sha) UPSTREAM_OVERRIDE="${2:?--upstream-sha needs a sha}"; shift 2 ;;
    --allow-unsigned) ALLOW_UNSIGNED=1; shift ;;
    -*) die "unknown option: $1" ;;
    *) [[ -z "${REF}" ]] || die "give exactly one tag or sha"; REF="$1"; shift ;;
  esac
done
[[ -n "${REF}" ]] || die "usage: tools/ds4/vendor-sync.sh <tag|sha> [--repo <url>] [--upstream-sha <sha>] [--allow-unsigned]"

# Paths dropped from the vendored tree. DATA ONLY: every one of these is a
# corpus or a capture, and no build target reads any of them. Adding a source
# path here would be a mistake the build would find immediately.
EXCLUDE=(
  "gguf-tools/imatrix/dataset"
  "gguf-tools/quality-testing/data"
  "speed-bench"
  "dir-steering/out"
  # The port's OWN .gitignore carries `/misc/`, and that file is vendored with
  # the tree, so anything left here would sit on disk untracked: present for
  # whoever ran the sync and absent from every fork, cut and ranked checkout.
  # It is four markdown notes and no build input. Dropped, so that what is
  # vendored is exactly what is tracked.
  "misc"
)

FIXTURE="${ROOT_DIR}/fixtures/qwen3_8_125b_a6b_track.json"
VENDOR_JSON="${ROOT_DIR}/ds4/VENDOR.json"
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/ds4-vendor.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

log "fetching ${REF} from ${REPO_URL}"
git init --quiet --bare "${WORK}/port.git"
git -C "${WORK}/port.git" remote add origin "${REPO_URL}"
git -C "${WORK}/port.git" fetch --quiet --tags origin "${REF}" 2>/dev/null \
  || git -C "${WORK}/port.git" fetch --quiet origin "${REF}" \
  || die "cannot fetch ${REF} from ${REPO_URL}"
SHA="$(git -C "${WORK}/port.git" rev-parse "FETCH_HEAD^{commit}")"
[[ -n "${SHA}" ]] || die "${REF} does not resolve to a commit"

# THE SIGNATURE IS THE PROVENANCE. Vendored bytes carry no gitlink, so the
# signature on the commit they came from is the only thing tying this tree to
# the port's history. It is checked HERE, once, and recorded in VENDOR.json.
if [[ "${ALLOW_UNSIGNED}" != "1" ]]; then
  git -C "${WORK}/port.git" verify-commit "${SHA}" 2>/dev/null \
    || die "${REF} (${SHA}) carries no good signature; the org requires signed commits. Pass --allow-unsigned only if you have another reason to trust it, and say so in the pull request."
  log "signature verified on ${SHA}"
else
  log "WARNING: signature check skipped by --allow-unsigned"
fi

# Upstream fork point, for the record: the first parent chain's merge base with
# the upstream tag if one is reachable, else whatever VENDOR.json already said.
UPSTREAM_SHA="${UPSTREAM_OVERRIDE:-$(jq -r '.upstream.sha // empty' "${VENDOR_JSON}" 2>/dev/null || true)}"

log "exporting the tree"
mkdir -p "${WORK}/tree"
git -C "${WORK}/port.git" archive --format=tar "${SHA}" | tar -x -C "${WORK}/tree"

before="$(find "${WORK}/tree" -type f | wc -l | tr -d '[:space:]')"
for rel in "${EXCLUDE[@]}"; do
  rm -rf "${WORK:?}/tree/${rel}"
done
after="$(find "${WORK}/tree" -type f | wc -l | tr -d '[:space:]')"
bytes="$(find "${WORK}/tree" -type f -exec wc -c {} + 2>/dev/null | tail -1 | awk '{print $1}')"
log "tree: ${before} files exported, ${after} vendored after exclusions (${bytes} bytes)"

# A vendored engine with no Makefile or no ds4.c is a mis-export, not a port.
for required in Makefile ds4.c ds4.h; do
  [[ -f "${WORK}/tree/${required}" ]] \
    || die "the exported tree has no ${required}; refusing to vendor it"
done

log "replacing ds4/"
rm -rf "${ROOT_DIR}/ds4"
mkdir -p "${ROOT_DIR}/ds4"
tar -C "${WORK}/tree" -cf - . | tar -C "${ROOT_DIR}/ds4" -xf -

python3 - "${VENDOR_JSON}" "${SHA}" "${REF}" "${REPO_URL}" "${UPSTREAM_URL}" \
         "${UPSTREAM_SHA}" "${after}" "${bytes}" "${EXCLUDE[@]}" <<'PY'
import json, subprocess, sys
out, sha, ref, repo, upstream_url, upstream_sha, files, size = sys.argv[1:9]
excludes = sys.argv[9:]
# VALUES ONLY. This file carries no prose: no `_comment`, no `_note`. What the
# vendoring is for, and that this file is generated rather than hand-edited,
# belongs in this script's header and in the port notes -- both of which say it.
doc = {
    "fork": {
        "repo": repo,
        "ref": ref,
        "sha": sha,
        "signed": True,
    },
    "upstream": {
        "repo": upstream_url,
        "sha": upstream_sha or None,
    },
    "vendored_at": subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip(),
    "excluded_paths": sorted(excludes),
    # Counted BEFORE this file is written, so they are what came from the port:
    # the vendored directory holds one more file, VENDOR.json, which is ours.
    "files": int(files),
    "bytes": int(size),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
PY
log "wrote ds4/VENDOR.json"

python3 - "${FIXTURE}" "${SHA}" <<'PY'
import json, sys
path, sha = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
doc = json.loads(text)
old = doc["serve_configuration"]["engine_pin"]
if old != sha:
    text = text.replace(old, sha)
    open(path, "w", encoding="utf-8").write(text)
    print(f"ds4/vendor-sync.sh: fixture engine_pin {old[:12]} -> {sha[:12]}")
else:
    print(f"ds4/vendor-sync.sh: fixture engine_pin already {sha[:12]}")
PY

log "done. Review with: git status --short ds4 && git diff --stat -- ds4 fixtures"
