#!/usr/bin/env bash
#
# Trip on the named ways a workflow in this repository could take a credential
# or ship a file out of the runner.
#
# This repository can receive organizer-material fixtures, so engine CI is
# build-and-test only. Two properties carry that: a job that holds no
# credential cannot reach hidden material, and a job that uploads nothing
# cannot leak what it does hold. Both are easy to lose by accident -- someone
# adds a debug artifact upload, or wires a submodule checkout token -- so they
# are checked instead of trusted.
#
# WHAT THIS IS, PRECISELY: a NAMED-PATTERN TRIPWIRE, not a proof of
# confinement. It fails the run on the patterns listed below and on nothing
# else. A workflow can still reach the network in ways no pattern here names --
# a `run:` step that curls, a third-party action that uploads on its own, an
# `actions/cache` entry used as a side channel. Those are review's job. This
# scan is the floor under review, so that the common accidental regressions
# cannot land silently.
#
# Detected:
#   - a `secrets` expression reference: `secrets.NAME`, `secrets['NAME']`, or
#     `toJSON(secrets)` (the whole map at once);
#   - `secrets: inherit` on a reusable-workflow call;
#   - any `write` permission, whether blanket (`permissions: write-all`) or a
#     single scope -- `id-token: write` is the OIDC-credential vector and is in
#     the scope list;
#   - an artifact upload, by action reference or by step expression, ANYWHERE
#     EXCEPT the one upload the ranked pipeline exists to produce (see below).
#
# THE ONE SANCTIONED UPLOAD. `.github/workflows/benchmark.yml` is the ranked
# pipeline, not engine CI, and its whole product is the score file Yukon reads
# back: benchmark.json's `scoreArtifact` names
# `benchmark-results-<run_id>/score.json`. Banning it outright would ban the
# track. So the blanket ban is NARROWED rather than lifted: uploads are still a
# hard failure everywhere else in .github/, and in that one file EXACTLY ONE
# upload may exist and it is itself checked -- pinned actions/upload-artifact
# action, `score.json` and nothing else, and `if-no-files-found: error` so a
# refused run cannot publish an empty artifact. A benchmark.yml that uploads
# anything wider, or that adds a second upload of any shape, fails this scan
# exactly as any other file would.
#
# The scan lives here rather than inline in the workflow so that the patterns
# it looks for are not themselves inside the scanned directory. It covers all
# of `.github/`, not just `.github/workflows`, so a composite action added
# alongside them is scanned too.
#
# Usage:  tools/ci-workflow-egress-scan.sh [<dir>]   (default .github)
# Exit:   0 no pattern matched, 1 one did.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${1:-${repo_root}/.github}"

if [[ ! -d "${dir}" ]]; then
  echo "ci-workflow-egress-scan: no such directory: ${dir}" >&2
  exit 2
fi

# Assembled at runtime for the reason in the header comment: a literal copy of
# these strings in a scanned file would be indistinguishable from a real hit.
#
# The reference pattern is deliberately NOT anchored to `${{ ... }}`: the same
# name is reachable from a composite action's `inputs` wiring and from a
# reusable-workflow `with:` block, and anchoring to the expression braces
# misses both.
secret_ref='secrets[.[(]'
secret_json='toJSON\([[:space:]]*'"secrets"
secret_inherit="secrets"':[[:space:]]*inherit'

# Blanket, and then per-scope. The scope list is GitHub's full set, so a new
# `write` scope is caught without this list having to guess at it.
perm_write_all='permissions:[[:space:]]*write'
perm_write_scope='^[[:space:]]+(actions|attestations|checks|contents|deployments|discussions|id-token|issues|models|packages|pages|pull-requests|repository-projects|security-events|statuses):[[:space:]]*write'

upload_action='actions/upload-'"artifact"
upload_expr='upload-'"artifact"

# The single file allowed to carry an upload, and what its upload must be.
ranked_workflow="${dir}/workflows/benchmark.yml"

status=0

if grep -rnE "${secret_ref}|${secret_json}|${secret_inherit}" "${dir}"; then
  echo "::error::a workflow references a repository secret; engine CI must hold no credential" >&2
  status=1
fi

if grep -rnE "${perm_write_all}|${perm_write_scope}" "${dir}"; then
  echo "::error::a workflow grants a write permission; engine CI's GITHUB_TOKEN must stay read-only" >&2
  status=1
fi

# Hits outside the ranked workflow are the original hard failure. `awk` rather
# than `grep -v` so the exemption is anchored to the start of grep's `path:`
# prefix and cannot be satisfied by the path appearing anywhere in a line.
upload_hits="$(grep -rnE "${upload_action}|${upload_expr}" "${dir}" || true)"
other_hits="$(printf '%s\n' "${upload_hits}" | awk -v p="${ranked_workflow}:" 'length($0) && index($0, p) != 1')"
if [[ -n "${other_hits}" ]]; then
  printf '%s\n' "${other_hits}"
  echo "::error::a workflow outside the ranked pipeline uploads an artifact; engine CI must ship nothing out of the runner" >&2
  status=1
fi

# Inside the ranked workflow the upload is allowed but not unexamined: it has
# to be THE ONE pinned action step, uploading score.json only, refusing an empty
# result. A missing benchmark.yml is fine here (nothing to upload, nothing to
# check); an upload in it that fails any of the four checks below is not.
#
# WHY THE COUNT COMES FIRST, AND WHY THE REST IS PER-STEP. Three independent
# "does the file contain X" greps are AT-LEAST-ONE tests, and a file can satisfy
# all three while also carrying a SECOND upload step that is unpinned, points at
# the staged tapes, and sets no if-no-files-found -- the conforming step answers
# every grep and the added one rides along unexamined. That is precisely the
# regression this whole scan exists to catch, so:
#
#   1. exactly ONE upload reference may exist in the file at all, and
#   2. the three property checks are evaluated INSIDE that one step's own block,
#      not file-wide, so a `path: score.json` belonging to some other step
#      cannot vouch for an upload that points somewhere else.
#
# Comments are stripped before counting: naming the action in prose is not an
# upload, but every occurrence in the YAML itself is one.
if [[ -f "${ranked_workflow}" ]] && grep -qE "${upload_expr}" "${ranked_workflow}"; then
  ranked_rel="${ranked_workflow#"${repo_root}"/}"
  ranked_yaml="$(sed 's/#.*$//' "${ranked_workflow}")"

  # Occurrences, not matching lines: two references on one line are still two.
  upload_refs="$(printf '%s\n' "${ranked_yaml}" | grep -oE "${upload_expr}" | grep -c . || true)"
  uses_refs="$(printf '%s\n' "${ranked_yaml}" | grep -cE "uses:[[:space:]]*${upload_action}@" || true)"

  if [[ "${upload_refs}" -ne 1 || "${uses_refs}" -ne 1 ]]; then
    echo "::error::${ranked_rel} carries ${upload_refs} artifact-upload reference(s) (${uses_refs} of them a \`uses:\` step); exactly one is allowed -- the pinned score upload. A second upload step is an egress path no matter how well-formed the first one is." >&2
    status=1
  elif ! grep -qE "uses:[[:space:]]*${upload_action}@[0-9a-f]{40}([[:space:]]|\$)" "${ranked_workflow}"; then
    echo "::error::${ranked_rel} uploads an artifact without a 40-hex-pinned ${upload_action} reference" >&2
    status=1
  else
    # Bound the property checks to the step that actually holds the `uses:`.
    # No ERE interval quantifiers in here: awk implementations disagree about
    # them, and the 40-hex pin is already grep's job above.
    step_verdict="$(printf '%s\n' "${ranked_yaml}" | awk -v act="${upload_action}" '
      { line[NR] = $0 }
      /^[[:space:]]*-[[:space:]]/ { marker[NR] = 1 }
      index($0, "uses:") && index($0, act "@") { uses_line = NR }
      END {
        if (uses_line == 0) { print "no-uses"; exit }
        start = 0
        for (i = uses_line; i >= 1; i--) if (marker[i]) { start = i; break }
        if (start == 0) { print "no-step"; exit }
        end = NR
        for (i = uses_line + 1; i <= NR; i++) if (marker[i]) { end = i - 1; break }
        for (i = start; i <= end; i++) {
          if (line[i] ~ /^[[:space:]]+path:[[:space:]]*score\.json[[:space:]]*$/) has_path = 1
          if (line[i] ~ /^[[:space:]]+if-no-files-found:[[:space:]]*error[[:space:]]*$/) has_inff = 1
        }
        if (!has_path) { print "path"; exit }
        if (!has_inff) { print "if-no-files-found"; exit }
        print "ok"
      }
    ')"
    case "${step_verdict}" in
      ok) : ;;
      path)
        echo "::error::${ranked_rel}'s upload step does not have exactly 'path: score.json' in its own block; the sanctioned upload is the score file and nothing else" >&2
        status=1
        ;;
      if-no-files-found)
        echo "::error::${ranked_rel}'s upload step does not set 'if-no-files-found: error' in its own block; a refused run must publish no artifact instead of an empty one" >&2
        status=1
        ;;
      *)
        echo "::error::${ranked_rel}'s upload could not be resolved to a single step block (${step_verdict}); refusing rather than guessing which upload was checked" >&2
        status=1
        ;;
    esac
  fi

  if [[ ${status} -eq 0 ]]; then
    echo "ci-workflow-egress-scan: ${ranked_rel} carries exactly one artifact upload, and it is pinned, score.json-only, and errors on no files"
  fi
fi

if [[ ${status} -eq 0 ]]; then
  echo "ci-workflow-egress-scan: no tripwire pattern matched in ${dir#"${repo_root}"/} (secret reference, secret inheritance, write permission, artifact upload outside the ranked score artifact)"
fi

exit ${status}
