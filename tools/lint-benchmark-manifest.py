#!/usr/bin/env python3
"""Lint benchmark.json -- the Yukon track manifest for this repository.

Yukon reads benchmark.json to import the track, overlay a submission's editable
surface onto a trusted checkout, and dispatch setup / pre-submit / benchmark. Every
one of those steps fails at RUN time on a manifest that is merely wrong at REST, so
this linter turns the rest-state properties into a check that runs in CI.

What it asserts, in order:

  1. schema keys        every key the live manifest carries is present, with the
                        right JSON type.
  2. editable paths     every editablePaths / optionalEditablePaths entry EXISTS in
                        the tree, optionalEditablePaths is a subset of editablePaths,
                        and no entry is a duplicate or is nested inside another.
  3. gitlink exclusion  no editable entry covers benchd/, .gitmodules, benchd.pin or
                        benchd-bin/ at ANY prefix depth, and tools/fetch-benchd.sh
                        exists.
                        LOAD-BEARING: benchd is the measurement harness --
                        a PREBUILT benchd resolved from the bench repository's
                        dist channel and verified against that channel's own
                        benchd.manifest.json -- and an editable entry over the
                        resolver or the resolved binary would let a submission
                        repoint the thing that scores it. The submodule and pin
                        spellings are kept so a reintroduced gitlink or a planted
                        pin file is covered on arrival.
  3b. byte budget       every editableSurfaceByteBudget cap is a positive integer, the
                        per-file cap can bind, and the enforcer's fallback constants
                        still equal the declared caps (manifest/enforcer drift).
  4. commands           setup / preSubmit / benchmark commands are ["bash","-c",...],
                        and every repo-relative script token in them exists and is
                        executable. benchmarkCommand must set MLXFAST_SCORE_PATH equal
                        to scorePath (the facade's --local-iterate default differs).
  5. contractPath       exists, parses as JSON, and its track_id matches trackId.
  5b. paired baseline   no golden carries benchmark.baseline_prefill_seconds_per_token
                        or benchmark.baseline_decode_seconds_per_token, and the
                        contract names a 40-hex baseline_reference_commit.
                        LOAD-BEARING: the ranked path measures its serial control
                        as a LEG, on the organizer-staged reference tree, on the
                        same box, in the same job. A stored pair is a denominator
                        captured on another machine, so it is refused at rest
                        here rather than read at score time.
  6. scoring constants  match the PINNED constants for this manifest's OWN trackId
                        (EXPECTED_SCORING_BY_TRACK -- a per-track registry, not one
                        global expectation: qwen3.8-27b-mtp-v1 and
                        qwen3.8-125b-a6b-cuda-v1 are ruled differently and neither's
                        pins apply to the other), and agree with the contract
                        fixture's scoring_semantics where both state a value. A
                        manifest value ruled AHEAD of the pinned benchd source
                        (this recurs at EVERY pairs ruling: pairsPerCohort=2 was
                        ruled ahead of the then-pinned compiled
                        PAIRS_PER_COHORT_TARGET=4 until the 2026-08-24
                        gitlink-advance brought the pin into agreement, and
                        pairsPerCohort=4 is ruled ahead of it again under the
                        2026-08-26 ruling until the channel tip carries it) is
                        NOT machine-cross-checked against benchd here -- benchd is a
                        prebuilt binary now, not a source tree this linter could
                        grep, so a check of that shape has nowhere to read from. The honest, load-bearing
                        instrument for that class of drift is
                        docs/qwen38-125b-a6b-port-notes.md (benchmark.json
                        and its contract fixture carry values only, no prose
                        fields, per David's 2026-08-24 ruling), not a linter
                        check.
  7. runner             runner.workflow resolves to a real file under
                        .github/workflows/.

Usage:  python3 tools/lint-benchmark-manifest.py [--repo-root DIR] [--manifest PATH]
                                                [--gitlink-targets require|report]
Exit:   0 all checks pass, 1 one or more failures (each printed with a FAIL prefix).

--gitlink-targets report downgrades a command target that is missing ONLY
because its submodule is not checked out to an UNVERIFIED line instead of a
FAIL. NO COMMAND TARGET IS INSIDE A SUBMODULE ANY MORE: benchd stopped being a
source submodule when it became a pinned prebuilt, and every command target now
lives in this repository (./setup.sh, ./tools/fetch-benchd.sh,
./tools/qwen38-125b-a6b-measure-and-score.sh), so the downgrade currently applies to
nothing and all three commands are fully verified in CI. The flag is kept for
the next submodule, not removed as dead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# --- per-track scoring-constant registry ------------------------------------
#
# RE-AIMED 2026-08-24 (reviewer blocker on the qwen3.8-125b-a6b-cuda-v1 manifest
# PR, orchestrator-adopted): this used to be ONE global EXPECTED_SCORING dict,
# hardcoded to the qwen3.8-27b-mtp-v1 ruling, that every manifest's `scoring`
# block was checked against regardless of track. That was never actually
# universal -- it just had exactly one registrant -- and it broke the instant
# a second, honestly differently-scored track (qwen3.8-125b-a6b-cuda-v1's
# batched-cohort composite, vs. qwen's per-prompt median) existed to check.
#
# THE FIX IS THE STRUCTURE, NOT A WEAKER CHECK: each track pins its own
# constants under its own trackId key below, and check_scoring looks up the
# CURRENT manifest's trackId. A track with a registered entry is checked
# EXACTLY as strictly as qwen's ever was -- the qwen3.8-27b-mtp-v1 entry below
# is byte-for-byte the content of the old global EXPECTED_SCORING dict, so
# every qwen3.8-27b-mtp-v1 manifest that passed this check before passes it
# identically now (verified: re-running this linter against a manifest with
# trackId=qwen3.8-27b-mtp-v1 and the old field values exercises the exact same
# key/value assertions, in the exact same fail/ok wording, as before this
# change). A track with NO registered entry is not silently waved through --
# see check_scoring's `expected is None` branch, which prints an explicit
# "no registered pin set" line rather than staying silent, so a future new
# track shows up as a visible gap instead of a quiet pass.
#
# QWEN3.8-27B-MTP-V1 sources, in-tree or in the pinned benchd source:
#   floor / ceiling / aggregation / median rule / pairs per prompt
#       fixtures/qwen3_8_27b_mtp_track.json  -> scoring_semantics
#   timed_mode "free_run_v1_1" and the never-compare rule
#       mlxfast-bench crates/bench-protocol/PROTOCOL.md -> "v1.1 additive extension"
#       docs/PROTOCOL-v1.1.md section 5 (hard rule), SIGNED 2026-08-17
#   decodeTokens N = 128
#       docs/PROTOCOL-v1.1.md open question 3, "RULED: N = 128"
_QWEN_MTP_V1_SCORING = {
    "mode": "qwen-mtp-paired-decode-only",
    "timedMode": "free_run_v1_1",
    "decodeTokens": 128,
    "mtpMaxDraftDepth": 8,
    "mtpEmptyDraftRoundsLegal": True,
    "aggregation": "median_of_per_prompt_raw_serial_relative_speedup",
    "medianRule": "even_n_mean_of_two_central_order_statistics",
    "scoreAnchor": "serial = 1.0",
    "noopReferenceRole": "informational_diagnostic_not_scored",
    "decodeSpeedupFloor": 0.90,
    "decodeSpeedupCeiling": 5.0,
    "pairsPerPrompt": 1,
    "minPairsPerPrompt": 1,
    "tokenFidelityGate": "trusted-sequential-reverification-exact-token-match",
    "tokenFidelityGateStatus": "implemented",
}

# QWEN3.8-125B-A6B-CUDA-V1 sources -- each value pinned here (not read from
# the fixture at check time, the same posture qwen's dict above always used)
# because each is either David's own ruling or a benchd source constant this
# repository's manifest independently pins to and documents:
#   mode
#       mlxfast-bench crates/benchd/src/overlay.rs `SCORING_MODE`, which is
#       the same string as measure_job.rs `MEASURE_JOB_MODE`: benchd's
#       SINGLE-STREAM paired regime name. It replaced
#       `COHORT_MEASURE_JOB_MODE` on 2026-08-27, when David ruled this track
#       single-stream ("Single-stream only": composite over a paired
#       serial-against-MTP single-stream series, scored_batch_size 1, and no
#       ContinuousBatchingV2 adaptation). Both names are read off benchd
#       source, never invented here -- an earlier manifest revision carried an
#       invented label benchd never emits, which is why this field is pinned
#       at all. A mismatch here is always an authoring error, never a
#       legitimate temporary state, so it has NO caveat-based warn path.
#   scoredBatchSize = 1
#       RULED by David 2026-08-27 with the mode above. THIS IS A
#       RULED-AHEAD-OF-PIN VALUE, like pairsPerCohort below: at the published
#       channel tip `ScoredBatchPoint::certify` certifies B = 8 only, so a
#       fixture declaring 1 is REFUSED at that width certification until the
#       bench lane lands the single-stream regime. The refusal is
#       fail-closed and correct; this repository declares the ruled shape and
#       does not work around it. docs/participant-contract.md section 5.1 and
#       section 11.4 carry the same statement.
#   scoredBatchSize / kvBackend / decodeSpeedupFloor / decodeSpeedupCeiling /
#   scoredExponents
#       benchmark.json scoring.* (this repo's own manifest); each field's
#       citation to the ruling and/or the benchd constant it certifies
#       against (ScoredBatchPoint::certify / SCORED_BATCH_SIZE_B8,
#       ScoredExponents::certify, kv_backend) lives in
#       docs/participant-contract.md section 5, not in the manifest itself
#       (benchmark.json and its contract fixture carry values only, no prose
#       fields, per David's 2026-08-24 ruling -- see
#       docs/qwen38-125b-a6b-port-notes.md for the full split).
#   pairsPerCohort = 4
#       RULED 4 by David 2026-08-26, verbatim: "you run it using 4 pairs
#       instead of 2 of 8 batches" -- 8 prompts x 4 pairs is the
#       challenger-grade sample mass the ruling buys. SUPERSESSION CHAIN, each
#       link superseding the one above it:
#         1. batch-8 brief D2 -- default 4;
#         2. David 2026-08-24 ("do 2") -- RULED 2, landed in benchd as commit
#            bb1a6216655912b8a57967bb9cd45cff973a82df, merged PR #184 at
#            047e21833a66264310307e1cb86ae3a290b0fc27 on the
#            qwen3.8-125b-a6b-cuda-v1 release branch;
#         3. David 2026-08-26 -- RULED 4 (this value), returning to the
#            brief's sample count on sample-mass grounds.
#       CHANNEL AGREEMENT WAS A TWO-PHASE LANDING, and it has landed: the
#       benchd side (PAIRS_PER_COHORT_TARGET 2 -> 4) merged and PUBLISHED, so
#       the served channel now compiles `PAIRS_PER_COHORT_TARGET: usize = 4`
#       (dist channel qwen3.8-125b-a6b-v1, source_commit 8439d6fe, verified
#       2026-08-30). An OFFICIAL run declaring target_pairs=4 is accepted; the
#       earlier ruled-ahead-of-pin state (the channel compiling 2 and refusing
#       4) is closed. See docs/qwen38-125b-a6b-port-notes.md
#       section 3 (which this dict's value must always match) for the full citation
#       chain. NOT machine-cross-checked against the pinned benchd source here
#       -- CI runs this linter without a benchd checkout, so that check could
#       only ever run locally; docs/qwen38-125b-a6b-port-notes.md remains
#       the load-bearing, always-visible instrument for any future drift of
#       this kind.
#   minPairsPerCohort = 4
#       ENFORCED AT THE PIN, same as pairsPerCohort. benchd refuses an OFFICIAL
#       batched cohort run whose min_pairs != PAIRS_PER_COHORT_TARGET, by name,
#       at the same pre-GPU seam as the target refusal (--local-dev still
#       explores other floors). Before that gate landed, benchd's only floor
#       rule was the parse-time `min_pairs <= target_pairs`, so a run declaring
#       min 2 / target 4 passed every trusted-side check and then published a
#       median over half the ruled support; the ruled floor rode entirely on
#       the wrapper's argv. It no longer does.
#
#       WHAT THIS LINTER ITSELF CHECKS, stated exactly, because the two are
#       easy to conflate: it pins the value of scoring.minPairsPerCohort in
#       benchmark.json against the registry below. It does NOT read
#       tools/qwen38-125b-a6b-measure-and-score.sh, so it does not compare the manifest
#       against the wrapper's actual --min-pairs literal -- a wrapper edited to
#       --min-pairs 2 while this manifest still said 4 would not be caught HERE.
#       That drift is caught at the pin instead: benchd refuses the run. The
#       wrapper's --min-pairs 4 is a belt-and-suspenders DECLARATION of the
#       ruled floor, and the wrapper lives under tools/ -- organizer-controlled,
#       outside editablePaths -- so a submission cannot rewrite it either way.
_QWEN38_125B_A6B_CUDA_V1_SCORING = {
    "mode": "qwen-native-mtp-paired-decode-only",
    "scoredBatchSize": 1,
    "kvBackend": "contiguous",
    "decodeSpeedupFloor": 0.90,
    "decodeSpeedupCeiling": 5.0,
    "scoredExponents": {"prefillGainExponent": 0.25, "decodeGainExponent": 0.75},
    "pairsPerCohort": 4,
    "minPairsPerCohort": 4,
}

EXPECTED_SCORING_BY_TRACK = {
    "qwen3.8-27b-mtp-v1": _QWEN_MTP_V1_SCORING,
    "qwen3.8-125b-a6b-cuda-v1": _QWEN38_125B_A6B_CUDA_V1_SCORING,
}

# Tracks whose scoring.seriesNote must document non-comparability against a
# PRIOR series on the SAME leaderboard family. This was written for exactly
# one fact: qwen3.8-27b-mtp-v1's v1.1 free-run regime is not comparable to the
# earlier qwen3.6-27b-mtp-v1 v1 teacher-forced board (frontier 1.376). That is
# a property of qwen's specific history, not a universal manifest
# requirement -- qwen3.8-125b-a6b-cuda-v1 has no earlier gemma4 series on this
# leaderboard family to be confused with, so it has nothing to disclaim and
# is correctly ABSENT from this dict rather than given an empty or vacuous
# entry.
#
# NOT RETARGETED by the 2026-08-24 manifest-notes-strip (David: benchmark.json
# and its contract fixture carry values only, no prose fields -- see
# docs/qwen38-125b-a6b-port-notes.md). This needle check is scoped entirely
# to qwen3.8-27b-mtp-v1's OWN manifest, a different repository this PR does
# not own; qwen3.8-125b-a6b-cuda-v1 has no entry here and therefore never
# exercises the scoring.seriesNote read below against THIS repository's
# manifest (confirmed: benchmark.json carries no seriesNote key at all, and
# the check prints "not required" for this trackId). A prose-carrying field
# genuinely still needed by an in-scope manifest is a live enforcement
# mechanism, not a stray note, so it stays as-is rather than being stripped
# or redirected to a doc it does not apply to.
SERIES_NOTE_REQUIRED_TRACKS = {
    "qwen3.8-27b-mtp-v1": ("NEW SERIES", "qwen3.6", "1.376", "never"),
}

# scoring key in benchmark.json -> scoring_semantics key in the contract fixture.
# Only keys BOTH files state; a disagreement between them is the drift this catches.
# Track-agnostic by construction (it only ever compares keys present in BOTH
# the manifest's own `scoring` block and its own contract's `scoring_semantics`
# block), so this needed no change for qwen3.8-125b-a6b-cuda-v1's different field
# names -- it simply finds zero shared keys with a track that does not use
# qwen's naming, and reports that as vacuously agreeing rather than failing.
CONTRACT_SCORING_MIRROR = {
    "aggregation": "aggregation",
    "medianRule": "median_rule",
    "scoreAnchor": "score_anchor",
    "decodeSpeedupFloor": "floor",
    "decodeSpeedupCeiling": "ceiling",
    "pairsPerPrompt": "pairs_per_prompt",
}

REQUIRED_KEYS = {
    "schemaVersion": int,
    "name": str,
    "trackId": str,
    "description": str,
    "category": str,
    "direction": str,
    "editablePaths": list,
    "optionalEditablePaths": list,
    "editableSurfaceByteBudget": dict,
    "contractPath": str,
    "setupCommand": list,
    "preSubmitCommand": list,
    "benchmarkCommand": list,
    "runner": dict,
    "scoreArtifact": str,
    "scorePath": str,
    "scoring": dict,
    "leaderboard": dict,
    "staticReviewTrackId": str,
}

# Editable entries must never reach these, at any prefix depth.
#
# `benchd` and `.gitmodules` named the SOURCE SUBMODULE that used to measure a
# submission. That submodule is gone, and so is the ./benchd.pin file that
# replaced it -- benchd now ships as a PREBUILT binary resolved from the bench
# repository's dist channel into ./benchd-bin/ by tools/fetch-benchd.sh -- so
# the entry that carries the live property is `benchd-bin`: an editable entry
# over the resolved-binary directory lets a submission swap the bytes after the
# check. Exactly the hole the gitlink exclusion existed to close, moved.
#
# The three dead entries are KEPT deliberately. They cost nothing, and a future
# tree that reintroduces a benchd submodule (or any submodule) must not have to
# rediscover that its gitlink is unsafe to declare editable. Removing a guard
# because its target is currently absent is how the guard is missing the next
# time the target is present.
FORBIDDEN_EDITABLE = ("benchd", ".gitmodules", "benchd.pin", "benchd-bin")

# benchmark.json is the SINGLE SOURCE for the editable-surface byte caps: the
# pre-dispatch static-review gate reads them from here. The five manifest keys
# whose values must be positive integers.
BUDGET_CAP_KEYS = (
    "maxTotalBytes",
    "maxFileBytes",
    "maxGrowthBytes",
    "exemptPathMaxBytes",
    "exemptPathMaxFileBytes",
)

# THERE IS ONE ENFORCER OF THOSE CAPS: the shell one, below, the pre-dispatch
# static review (.github/scripts/submission-static-review-checks.sh). The Swift
# EditableSurfaceByteBudget.swift enforcer was a parity twin of it and was
# removed with the whole Apple-Metal/Swift seed package in the final de-Swift
# (Stage 5a); it was never in a live launch path here (no caller invoked it),
# so nothing on the scored/enforced surface changed when it went.
#
# The enforcer prefers the manifest and falls back to a compiled-in constant
# only for a contract that declares no caps, so that constant can drift away
# from the declaration. It once did: maxTotalBytes was raised in the manifest
# and left stale in the shell fallback, and this linter passed clean across the
# gap for a day. A shell fallback BELOW the declaration is a latent
# false-reject (static review refuses a submission the contract says fits); one
# ABOVE it admits bytes the contract does not. Neither direction is
# submission-reachable -- a submission cannot edit the enforcer -- so this is a
# consistency gate, not a hole being closed; the reason to hold it tight anyway
# is that "the enforcer agrees with the contract" is the property the
# single-source design is FOR, and a silent gap in it is how a cap ends up
# meaning two things.
SHELL_ENFORCER = os.path.join(".github", "scripts", "submission-static-review-checks.sh")

# `resolve_cap VAR_NAME CONTRACT_KEY FALLBACK`, one per line at column 0. Read
# from the real call sites rather than from a restated list, so a renamed key or
# a fourth cap cannot slip past by not being in a list here. The suite's
# companion assertion (drift/static-review still resolves three caps through
# resolve_cap) pins the COUNT; this pins the VALUES.
RESOLVE_CAP_RE = re.compile(r"^resolve_cap\s+(\w+)\s+(\w+)\s+([0-9]+)\s*$", re.MULTILINE)


class Linter:
    def __init__(
        self, root: str, manifest_path: str, gitlink_targets: str = "require"
    ) -> None:
        self.root = root
        self.manifest_path = manifest_path
        # "require": a command target inside a submodule must exist.
        # "report": a target that is missing ONLY because its submodule is not
        # checked out is reported as unverified instead of failing. This repo has
        # no submodules since benchd became a pinned prebuilt, so neither mode
        # currently downgrades anything and CI verifies all three commands; the
        # distinction is retained for the next submodule. The check is never
        # skipped, only named -- see the UNVERIFIED lines and the run summary.
        self.gitlink_targets = gitlink_targets
        self.failures: list[str] = []
        self.unverified: list[str] = []
        self.checks = 0

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"FAIL  {msg}")

    def ok(self, msg: str) -> None:
        self.checks += 1
        print(f"ok    {msg}")

    def unverifiable(self, msg: str) -> None:
        self.unverified.append(msg)
        print(f"UNVERIFIED  {msg}")

    def abspath(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def gitlink_paths(self) -> list[str]:
        """Submodule paths declared in .gitmodules, as repo-relative strings."""
        paths = []
        try:
            with open(self.abspath(".gitmodules"), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("path") and "=" in line:
                        paths.append(line.split("=", 1)[1].strip())
        except OSError:
            pass
        return paths

    def missing_because_gitlink_absent(self, rel: str) -> str | None:
        """The submodule path that explains a missing target, or None.

        A gitlink whose working tree was never initialised is an EMPTY
        directory. If the submodule IS checked out and the target is still
        missing, that is a real failure and this returns None.
        """
        for sub in self.gitlink_paths():
            if rel == sub or rel.startswith(sub.rstrip("/") + "/"):
                sub_abs = self.abspath(sub)
                if os.path.isdir(sub_abs) and not os.listdir(sub_abs):
                    return sub
        return None

    # -- 1 -----------------------------------------------------------------
    def check_schema(self, m: dict) -> None:
        for key, typ in REQUIRED_KEYS.items():
            if key not in m:
                self.fail(f"schema: missing required key {key!r}")
            elif not isinstance(m[key], typ):
                self.fail(
                    f"schema: key {key!r} is {type(m[key]).__name__}, expected {typ.__name__}"
                )
        for key in ("provider", "workflow"):
            if key not in m.get("runner", {}):
                self.fail(f"schema: missing runner.{key}")
        if "namespace" not in m.get("leaderboard", {}):
            self.fail("schema: missing leaderboard.namespace")
        if m.get("trackId") != m.get("staticReviewTrackId"):
            self.fail(
                f"schema: trackId {m.get('trackId')!r} != staticReviewTrackId "
                f"{m.get('staticReviewTrackId')!r}"
            )
        if not self.failures:
            self.ok(f"schema: all {len(REQUIRED_KEYS)} required keys present and typed")

    # -- 2 -----------------------------------------------------------------
    def check_editable_paths(self, m: dict) -> None:
        paths = m.get("editablePaths", [])
        missing = [p for p in paths if not os.path.exists(self.abspath(p))]
        for p in missing:
            self.fail(f"editablePaths: {p} does not exist in the tree")
        if not missing:
            self.ok(f"editablePaths: all {len(paths)} entries exist in the tree")

        dupes = {p for p in paths if paths.count(p) > 1}
        for p in sorted(dupes):
            self.fail(f"editablePaths: duplicate entry {p}")
        if not dupes:
            self.ok("editablePaths: no duplicate entries")

        # A path nested inside another editable path is redundant and makes the
        # byte budget and the overlay ambiguous about which rule applies.
        nested = []
        for p in paths:
            for q in paths:
                if p != q and p.startswith(q.rstrip("/") + "/"):
                    nested.append((p, q))
        for p, q in nested:
            self.fail(f"editablePaths: {p} is nested inside {q}")
        if not nested:
            self.ok("editablePaths: no entry nested inside another")

        optional = m.get("optionalEditablePaths", [])
        stray = [p for p in optional if p not in paths]
        for p in stray:
            self.fail(f"optionalEditablePaths: {p} is not in editablePaths")
        opt_missing = [p for p in optional if not os.path.exists(self.abspath(p))]
        for p in opt_missing:
            self.fail(f"optionalEditablePaths: {p} does not exist in the tree")
        if not stray and not opt_missing:
            self.ok(
                f"optionalEditablePaths: all {len(optional)} entries exist and are a "
                "subset of editablePaths"
            )

        exempt = m.get("editableSurfaceByteBudget", {}).get("exemptPaths", [])
        bad = [p for p in exempt if p not in paths]
        for p in bad:
            self.fail(f"editableSurfaceByteBudget.exemptPaths: {p} is not an editable path")
        if not bad:
            self.ok("editableSurfaceByteBudget.exemptPaths: subset of editablePaths")

    # -- shared path helpers (check 3, the gitlink exclusion) ----------------
    #
    # The gitlink guard (check 3) decides one question about each entry: does
    # this editable path reach something it must not. It has to be consistent
    # about what an entry SPELLS, about which entries it walks, and about which
    # spellings are not paths at all. The helpers below are that vocabulary.
    # (They were once shared with a trusted-scope overlap guard -- linter check
    # 3c -- removed 2026-08-30 per David's no-unnecessary-guards ruling.)

    def _same_file(self, a: str, b: str) -> bool:
        """True when two repo-relative paths are the same file on this filesystem.

        Both must exist; a path that does not resolve is handled by the folded
        string comparison instead.

        This arm is NOT available on the hosted lint runner for a wrong-case
        spelling: ubuntu's ext4 is case-SENSITIVE, so `sources/mlxfastcore` does
        not resolve there and only the folded string comparison binds. That is
        why the lexical arm must normalise -- see _normalize().
        """
        if not a or a in (".", "/"):
            return False
        try:
            return os.path.samefile(self.abspath(a), self.abspath(b))
        except OSError:
            return False

    @staticmethod
    def _editable_buckets(m: dict) -> tuple[tuple[str, list], ...]:
        """The three manifest keys an overlay writes from, as (name, entries).

        Used by check 3 (the gitlink exclusion) so it guards every entry an
        overlay can write from. exemptPaths belongs here: it exempts bytes
        from the code budget, not the path from the overlay.
        """
        return (
            ("editablePaths", list(m.get("editablePaths", []))),
            ("optionalEditablePaths", list(m.get("optionalEditablePaths", []))),
            (
                "editableSurfaceByteBudget.exemptPaths",
                list(m.get("editableSurfaceByteBudget", {}).get("exemptPaths", [])),
            ),
        )

    @staticmethod
    def _normalize(rel: str) -> str:
        """A repo-relative path reduced to the join of its non-empty segments.

        THE SINGLE REDUCTION. Every lexical comparison in check 3 runs on this
        form, so the string arm and the filesystem arm can no longer disagree
        about the same entry.

        The bug this closes: the lexical arm used to compare `entry.strip("/")`
        raw while _prefixes() silently dropped empty segments. `Sources//MLXFastCore`
        therefore folded to `sources//mlxfastcore`, matched neither
        `== "sources/mlxfastcore"` nor `startswith("sources/mlxfastcore/")`, and
        fell through to the samefile arm -- which rescued it ONLY because the
        path happened to resolve. Case-mangle it as well (`sources//mlxfastcore`)
        and on the case-sensitive hosted lint runner nothing bound at all:
        verified returning None there before this change.
        """
        return "/".join(p for p in rel.strip("/").split("/") if p not in ("", "."))

    @staticmethod
    def _illegal_editable_entry(entry: str) -> str | None:
        """Why `entry` is not a legal repo-relative editable path, or None.

        Mirrors the overlay's validity rule byte for byte
        (.github/scripts/overlay-editable-paths.sh:87,91-96), whose own comment
        says the two validators must not disagree about what a legal editable
        path is. Checked BEFORE the overlap arithmetic in BOTH guards, because
        every spelling here defeats that arithmetic rather than failing it:

          ''  '.'  './'   resolve to the repository ROOT, which contains every
                          trusted path -- and _prefixes() renders them as the
                          empty list, so no comparison happens at all;
          '/abs/path'     os.path.join(root, rel) DISCARDS root for an absolute
                          rel (:183-184), and strip('/') then re-roots the
                          entry under the repo, so an absolute path naming the
                          real Sources/MLXFastCore -- or the real benchd --
                          compared as neither equal nor same-file;
          ':pathspec'     git pathspec magic, not a path;
          'a/../b'        only caught downstream when it happens to resolve on
                          this filesystem.

        None of these is a live hole -- the overlay refuses all of them at run
        time before anything is written -- but the linter is the rest-state
        gate and must not be the layer that says yes.
        """
        if not entry or not entry.strip():
            return "is empty"
        if entry.startswith("/"):
            return "is an absolute path, not a repo-relative one"
        if entry.startswith(":"):
            return "is a pathspec, not a path"
        if "\\" in entry:
            return "contains a backslash"
        if "/./" in f"/{entry}/" or "/../" in f"/{entry}/":
            return "contains a '.' or '..' segment"
        return None

    # -- 3 -----------------------------------------------------------------
    def check_gitlink_excluded(self, m: dict) -> None:
        """No editable entry may cover the benchd gitlink or .gitmodules.

        Checked as a PREFIX relation in both directions: an entry equal to, inside,
        or containing a forbidden path all fail. "benchdx" must not trip it, so the
        containment test appends a separator. FORBIDDEN_EDITABLE covers both the
        live pin paths (benchd.pin, benchd-bin) and the retired submodule
        spellings (benchd, .gitmodules); see its definition.

        CASE-FOLDED, and separately checked by filesystem identity. The ranked box
        is macOS and APFS is case-INSENSITIVE by default, so an entry spelled
        "BENCHD.PIN" names the real pin; a byte comparison passes it and the
        overlay's rm -rf then replaces what decides which scorer runs. str.casefold() normalises
        the spellings ASCII folding reaches, and os.path.samefile() catches the
        ones it does not (Unicode folding, HFS+ decomposition) whenever the entry
        actually resolves on this filesystem.

        SHAPE FIRST. _illegal_editable_entry() runs before any of that, because
        the spellings that defeat this guard's arithmetic must be refused rather
        than reasoned about: an ABSOLUTE path naming the real pin is re-rooted
        under the repo by strip('/'), so it compares as neither equal, inside,
        containing nor same-file, and '' / '.' / './' name the repository root
        that CONTAINS it. Every such entry is refused by the one validity rule.

        The failing BUCKET is named: three buckets are walked and an entry alone
        does not say which one to fix.
        """
        buckets = self._editable_buckets(m)
        hits = []
        entry_count = 0
        illegal_count = 0
        for bucket, entries in buckets:
            for entry in entries:
                if not isinstance(entry, str):
                    continue
                entry_count += 1
                illegal = self._illegal_editable_entry(entry)
                if illegal is not None:
                    self.fail(
                        f"gitlink exclusion: {bucket} entry {entry!r} {illegal} -- an entry "
                        "this shape cannot be shown NOT to cover the pinned measurement "
                        "daemon, so it is refused rather than reasoned about"
                    )
                    illegal_count += 1
                    continue
                norm = self._normalize(entry)
                folded = norm.casefold()
                for forbidden in FORBIDDEN_EDITABLE:
                    folded_forbidden = self._normalize(forbidden).casefold()
                    if folded == folded_forbidden:
                        hits.append((bucket, entry, forbidden, "equals"))
                    elif folded.startswith(folded_forbidden + "/"):
                        hits.append((bucket, entry, forbidden, "is inside"))
                    elif folded_forbidden.startswith(folded + "/"):
                        hits.append((bucket, entry, forbidden, "contains"))
                    elif self._same_file(norm, forbidden):
                        hits.append((bucket, entry, forbidden, "resolves to"))
                    elif self._same_file(norm.split("/", 1)[0], forbidden):
                        hits.append((bucket, entry, forbidden, "is inside"))
        for bucket, entry, forbidden, how in hits:
            self.fail(
                f"gitlink exclusion: {bucket} entry {entry!r} {how} {forbidden!r} -- "
                "a submission must never be able to edit the pinned measurement daemon"
            )
        if not hits and not illegal_count:
            self.ok(
                "gitlink exclusion: no editable entry covers the benchd pin paths "
                f"({', '.join(FORBIDDEN_EDITABLE)}) ({entry_count} entries checked)"
            )

        # NON-VACUITY. The exclusion is only worth anything if the thing it
        # refuses to make editable actually exists in the tree. That used to be
        # `.gitmodules` (submodule era), then `benchd.pin` (sha-pin era). The
        # pin is RETIRED (David ruling 2026-08-27: benchd resolves from the
        # bench branch's dist channel, verified against the channel's
        # benchd.manifest.json, so measurement fixes ship bench-side with no
        # engine commit); the artifact carrying the same authority is now
        # tools/fetch-benchd.sh -- the channel constants and the manifest
        # verification live there, trusted-side. A tree where that script is
        # gone is a tree where this check guards nothing, and it must go red
        # rather than quietly pass. (The `benchd.pin` SPELLING stays excluded
        # above so a submission can never plant a pin file.)
        fetcher = self.abspath("tools/fetch-benchd.sh")
        if os.path.exists(fetcher):
            self.ok("tools/fetch-benchd.sh exists (benchd is a channel prebuilt, as the exclusion assumes)")
        else:
            self.fail(
                "tools/fetch-benchd.sh missing -- benchd resolves from the dist "
                "channel through that script; without it there is nothing binding "
                "which measurement binary runs"
            )

    # -- 3b ----------------------------------------------------------------
    def check_byte_budget(self, m: dict) -> None:
        """The byte caps are declared here and READ by the enforcer.

        This checks the manifest side: every cap is a positive integer, the
        per-file cap can bind, and the compiled-in fallback in the enforcer
        (used only for a contract that declares no caps) still agrees with what
        is declared -- .github/scripts/submission-static-review-checks.sh at
        pre-dispatch static review.
        """
        budget = m.get("editableSurfaceByteBudget", {})
        caps = {}
        for key in BUDGET_CAP_KEYS:
            value = budget.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                self.fail(
                    f"editableSurfaceByteBudget.{key}: {value!r} is not a positive integer -- "
                    "the enforcers read this manifest, so an unusable cap is a dispatch bug"
                )
            else:
                caps[key] = value
        if len(caps) != len(BUDGET_CAP_KEYS):
            return
        if caps["maxFileBytes"] > caps["maxTotalBytes"]:
            self.fail(
                f"editableSurfaceByteBudget: maxFileBytes ({caps['maxFileBytes']}) exceeds "
                f"maxTotalBytes ({caps['maxTotalBytes']}); the per-file cap can never bind"
            )
        elif caps["exemptPathMaxFileBytes"] > caps["exemptPathMaxBytes"]:
            # A MANIFEST-SHAPE check, not a resolution rule. The enforcers
            # deliberately resolve this pair without a relation guard (a
            # contract may declare only the aggregate and take the per-file
            # default), but THIS repository's manifest declares both, so a pair
            # that can never bind here is an authoring mistake worth naming.
            self.fail(
                f"editableSurfaceByteBudget: exemptPathMaxFileBytes "
                f"({caps['exemptPathMaxFileBytes']}) exceeds exemptPathMaxBytes "
                f"({caps['exemptPathMaxBytes']}); the exempt per-file cap can never bind"
            )
        else:
            self.ok(
                "editableSurfaceByteBudget: all five caps are positive integers and "
                "both per-file caps bind"
            )

        self._check_shell_fallbacks(caps)

    def _check_shell_fallbacks(self, caps: dict) -> None:
        """The static-review gate's resolve_cap fallbacks must equal the manifest.

        The static review is the one enforcer of these caps. See SHELL_ENFORCER
        for how a stale fallback silently changed a cap once.
        """
        shell = self.abspath(SHELL_ENFORCER)
        try:
            with open(shell, encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            self.fail(
                f"editableSurfaceByteBudget: cannot read the static-review enforcer: {exc}"
            )
            return

        sites = RESOLVE_CAP_RE.findall(source)
        if not sites:
            # VACUITY GUARD. Every failure below is "a call site disagrees", and
            # a regex that has stopped matching reports zero disagreements --
            # the same green a correct file gives. A file that still exists but
            # declares no cap the way this check can read is a broken check, not
            # a passing one, and it says so here rather than in a ranked run.
            self.fail(
                f"editableSurfaceByteBudget: found no `resolve_cap` call site in "
                f"{SHELL_ENFORCER} -- either the gate stopped resolving its caps or "
                "the call signature changed; this drift check would otherwise pass "
                "vacuously"
            )
            return

        drift = []
        for var, key, literal in sites:
            if key not in caps:
                # An enforcer resolving a key the contract never declares runs
                # on its fallback ALWAYS, which is the single-source design
                # inverted: the constant, not the manifest, is the cap.
                drift.append(
                    f"resolve_cap {var} {key} names a cap the manifest does not "
                    f"declare, so it can only ever use its fallback {literal}"
                )
                continue
            if int(literal) != caps[key]:
                drift.append(
                    f"resolve_cap {var} {key} falls back to {literal}, "
                    f"manifest {key} is {caps[key]}"
                )
        for item in drift:
            self.fail(f"editableSurfaceByteBudget drift: {item}")
        if not drift:
            self.ok(
                f"editableSurfaceByteBudget: the static-review gate's "
                f"{len(sites)} resolve_cap fallback(s) equal the declared caps"
            )

    # -- 4 -----------------------------------------------------------------
    def _check_command(self, m: dict, key: str) -> None:
        cmd = m.get(key)
        if not isinstance(cmd, list) or len(cmd) != 3 or cmd[0] != "bash" or cmd[1] != "-c":
            self.fail(f"{key}: expected [\"bash\", \"-c\", <script>], got {cmd!r}")
            return
        script = cmd[2]
        # Every ./-rooted token in the script is a repo-relative entry point.
        targets = [
            tok.strip("\"'")
            for tok in script.replace("&&", " ").split()
            if tok.startswith("./")
        ]
        if not targets:
            self.fail(f"{key}: script names no repo-relative entry point: {script!r}")
            return
        for t in targets:
            p = self.abspath(t)
            rel = t[2:] if t.startswith("./") else t
            if not os.path.exists(p):
                sub = self.missing_because_gitlink_absent(rel)
                if sub is not None and self.gitlink_targets == "report":
                    self.unverifiable(
                        f"{key}: target {t} lives in the {sub!r} submodule, which is "
                        "not checked out here -- existence and executability "
                        "UNVERIFIED (run this linter where the submodule is "
                        "initialised)"
                    )
                else:
                    self.fail(f"{key}: target {t} does not exist")
            elif not os.access(p, os.X_OK):
                self.fail(f"{key}: target {t} exists but is not executable")
            else:
                self.ok(f"{key}: target {t} exists and is executable")

    def check_commands(self, m: dict) -> None:
        for key in ("setupCommand", "preSubmitCommand", "benchmarkCommand"):
            self._check_command(m, key)

        # The facade defaults --local-iterate to score.local-iterate.json, so the
        # benchmark command MUST pin MLXFAST_SCORE_PATH or Yukon reads the wrong file.
        script = m.get("benchmarkCommand", [None, None, ""])[-1]
        want = f"MLXFAST_SCORE_PATH={m.get('scorePath')}"
        if want in script:
            self.ok(f"benchmarkCommand: pins {want} (matches scorePath)")
        else:
            self.fail(
                f"benchmarkCommand: must set {want}; the facade's --local-iterate "
                "default is score.local-iterate.json, which would not match scorePath"
            )

    # -- 5 -----------------------------------------------------------------
    def check_contract(self, m: dict) -> dict:
        rel = m.get("contractPath", "")
        path = self.abspath(rel)
        if not os.path.exists(path):
            self.fail(f"contractPath: {rel} does not exist")
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                contract = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.fail(f"contractPath: {rel} does not parse as JSON: {exc}")
            return {}
        self.ok(f"contractPath: {rel} exists and parses ({len(contract)} top-level keys)")

        if contract.get("track_id") != m.get("trackId"):
            self.fail(
                f"contractPath: track_id {contract.get('track_id')!r} != manifest "
                f"trackId {m.get('trackId')!r}"
            )
        else:
            self.ok(f"contractPath: track_id matches trackId ({m.get('trackId')})")

        pool = contract.get("timed_prompt_pool", [])
        if len(pool) == 8:
            self.ok("contractPath: timed_prompt_pool has 8 prompts (the median rule assumes even n)")
        else:
            self.fail(
                f"contractPath: timed_prompt_pool has {len(pool)} prompts, expected 8 "
                "(scoring.medianRule and pairsPerPromptNote both assume 8)"
            )
        return contract

    # -- 5b ----------------------------------------------------------------
    def check_paired_baseline(self, contract: dict) -> None:
        """No golden may carry a stored baseline pair; the fixture must name the
        reference commit instead.

        PAIRED, PER-BOX BASELINE (David 2026-09-08). The serial control is a LEG
        MEASURED ON THE BOX, on the organizer-staged reference tree, in the same
        job as the candidate. A stored pair is therefore not a stale number, it
        is a WRONG denominator: it was captured on another box, on another
        engine, at another temperature, and anything that reads it scores the
        candidate against a machine it never ran on. The pair is refused at rest
        here, so it cannot come back by a golden re-author.

        The check reads the FILES, not the fixture's copy of their digests: the
        field is a property of the golden's bytes.
        """
        banned = (
            "baseline_prefill_seconds_per_token",
            "baseline_decode_seconds_per_token",
        )

        # Every golden the fixture names, plus every *.json beside them. The
        # directory sweep is what catches a per-depth tape that is staged but
        # not yet pinned.
        rels: set[str] = set()
        for entry in contract.get("timed_prompt_pool", []):
            if isinstance(entry, dict) and isinstance(entry.get("r2_path"), str):
                rels.add(entry["r2_path"])
        spec = contract.get("live_golden_speculative", {})
        if isinstance(spec, dict):
            for entry in spec.values():
                if isinstance(entry, dict) and isinstance(entry.get("r2_path"), str):
                    rels.add(entry["r2_path"])
        for rel in sorted(rels):
            d = os.path.dirname(rel)
            if not os.path.isdir(self.abspath(d)):
                continue
            for name in os.listdir(self.abspath(d)):
                if name.endswith(".json"):
                    rels.add(os.path.join(d, name))

        if not rels:
            self.fail(
                "goldens: the contract names no golden file, so the baseline-pair "
                "check has nothing to read"
            )
            return

        carriers: list[str] = []
        for rel in sorted(rels):
            path = self.abspath(rel)
            if not os.path.exists(path):
                self.fail(f"goldens: {rel} is named by the contract but is not in the tree")
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    golden = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                self.fail(f"goldens: {rel} does not parse as JSON: {exc}")
                continue
            bench = golden.get("benchmark", {})
            if not isinstance(bench, dict):
                continue
            found = [f for f in banned if f in bench]
            if found:
                carriers.append(f"{rel} ({', '.join(found)})")

        if carriers:
            for c in carriers:
                self.fail(
                    f"goldens: {c} carries a stored baseline pair. The ranked path is "
                    "PAIRED with a per-box serial-control leg, so a stored pair is a "
                    "denominator measured on another machine. Strip the field and "
                    "re-pin the golden"
                )
        else:
            self.ok(
                f"goldens: none of the {len(rels)} golden file(s) carries "
                "benchmark.baseline_prefill_seconds_per_token or "
                "benchmark.baseline_decode_seconds_per_token (paired per-box baseline)"
            )

        # ANY TREE THAT CAN BE PROMOTED AS A REFERENCE MUST SPEAK THE PER-LEG
        # VERBS. benchd boots each leg's resident by running that leg's own
        # tools/serve-up.sh with --boot --spec ... --socket-out FILE, and ends it
        # with --stop --socket PATH. A tree whose serve script has only the
        # wrapper form cannot serve a leg at all, and the failure would land
        # inside the GPU window as an argument error.
        #
        # The check is at REST and on THIS tree, because that is where it can be
        # fixed. On a box the question does not arise: the workspace is pinned to
        # baseline_reference_commit and verified clean, so its serve script is
        # exactly that commit's.
        serve_up = self.abspath(os.path.join("tools", "serve-up.sh"))
        if not os.path.exists(serve_up):
            self.fail("serve-up: tools/serve-up.sh does not exist")
        else:
            with open(serve_up, encoding="utf-8") as fh:
                serve_src = fh.read()
            missing = [v for v in ("--boot", "--stop", "--socket-out", "--socket")
                       if v not in serve_src]
            if missing:
                self.fail(
                    f"serve-up: tools/serve-up.sh does not implement {', '.join(missing)}; "
                    "benchd boots each leg's resident with the per-leg verbs, so a tree "
                    "without them cannot serve a paired leg"
                )
            else:
                self.ok(
                    "serve-up: tools/serve-up.sh implements the per-leg verbs "
                    "(--boot/--stop with --socket-out/--socket)"
                )

        # The fixture must name the reference tree the serial-control leg runs
        # on. Without it the preflight has nothing to check the staged workspace
        # against, and any tree would pass as the reference.
        ref = contract.get("baseline_reference_commit")
        if isinstance(ref, str) and re.fullmatch(r"[0-9a-f]{40}", ref):
            self.ok(f"contractPath: baseline_reference_commit is a full sha ({ref[:12]})")
        else:
            self.fail(
                f"contractPath: baseline_reference_commit must be a 40-hex commit sha, "
                f"got {ref!r}; the serial-control leg has no reference tree to run on"
            )

    # -- 6 -----------------------------------------------------------------
    def check_scoring(self, m: dict, contract: dict) -> None:
        scoring = m.get("scoring", {})
        track_id = m.get("trackId", "")
        bad = False

        # PER-TRACK constant registry (see EXPECTED_SCORING_BY_TRACK's own
        # comment for why this replaced one global dict). A track with no
        # registered entry is a visible gap, not a silent pass: it prints an
        # explicit line saying so rather than skipping the check without a
        # trace.
        expected = EXPECTED_SCORING_BY_TRACK.get(track_id)
        if expected is None:
            self.ok(
                f"scoring: trackId {track_id!r} has no entry in EXPECTED_SCORING_BY_TRACK "
                "(tools/lint-benchmark-manifest.py) -- register one there the same way "
                "qwen3.8-27b-mtp-v1 and qwen3.8-125b-a6b-cuda-v1 are, rather than leave a "
                "new track's scoring constants unchecked indefinitely"
            )
        else:
            for key, exp_value in expected.items():
                actual = scoring.get(key)
                if actual != exp_value:
                    self.fail(
                        f"scoring.{key}: {actual!r}, expected {exp_value!r} per the pinned "
                        f"{track_id} scoring ruling"
                    )
                    bad = True
            if not bad:
                self.ok(
                    f"scoring: all {len(expected)} constants match the pinned "
                    f"{track_id} ruling"
                )

        # The series declaration must be legible in the file itself, not only here --
        # but ONLY for tracks that actually have a prior series on the same
        # leaderboard family to disclaim non-comparability against (see
        # SERIES_NOTE_REQUIRED_TRACKS's own comment).
        needles = SERIES_NOTE_REQUIRED_TRACKS.get(track_id)
        if needles is None:
            self.ok(
                f"scoring.seriesNote: not required for trackId {track_id!r} (no entry in "
                "SERIES_NOTE_REQUIRED_TRACKS -- no prior series on this leaderboard "
                "family to disclaim non-comparability against)"
            )
        else:
            note = scoring.get("seriesNote", "")
            for needle in needles:
                if needle.lower() not in note.lower():
                    self.fail(
                        f"scoring.seriesNote: must state {needle!r} -- the non-comparability "
                        f"of this board to the prior {track_id} series is a hard rule and has "
                        "to be documented in the manifest"
                    )
                    bad = True
            if not bad:
                self.ok("scoring.seriesNote: declares the new series and names what it is not comparable to")

        semantics = contract.get("scoring_semantics", {})
        if semantics:
            drift = False
            for mkey, ckey in CONTRACT_SCORING_MIRROR.items():
                if ckey in semantics and scoring.get(mkey) != semantics[ckey]:
                    self.fail(
                        f"scoring.{mkey} = {scoring.get(mkey)!r} disagrees with "
                        f"contract scoring_semantics.{ckey} = {semantics[ckey]!r}"
                    )
                    drift = True
            if not drift:
                self.ok(
                    f"scoring: agrees with contract scoring_semantics on all "
                    f"{len(CONTRACT_SCORING_MIRROR)} shared constants"
                )

        if scoring.get("decodeSpeedupFloor", 0) >= scoring.get("decodeSpeedupCeiling", 0):
            self.fail("scoring: decodeSpeedupFloor is not below decodeSpeedupCeiling")
        else:
            self.ok("scoring: floor < ceiling")

    # -- 7 -----------------------------------------------------------------
    def check_runner(self, m: dict) -> None:
        wf = m.get("runner", {}).get("workflow", "")
        rel = os.path.join(".github", "workflows", wf)
        if os.path.exists(self.abspath(rel)):
            self.ok(f"runner.workflow: {rel} exists")
        else:
            self.fail(f"runner.workflow: {rel} does not exist")

    def run(self) -> int:
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL  manifest: {self.manifest_path} does not parse: {exc}")
            return 1

        print(f"linting {self.manifest_path}")
        print(f"repo root {self.root}")
        print()
        self.check_schema(manifest)
        self.check_editable_paths(manifest)
        self.check_gitlink_excluded(manifest)
        self.check_byte_budget(manifest)
        self.check_commands(manifest)
        contract = self.check_contract(manifest)
        self.check_paired_baseline(contract)
        self.check_scoring(manifest, contract)
        self.check_runner(manifest)

        print()
        if self.unverified:
            print(f"{len(self.unverified)} check(s) UNVERIFIED (submodule not checked out):")
            for msg in self.unverified:
                print(f"  - {msg}")
        if self.failures:
            print(f"{len(self.failures)} FAILURE(S), {self.checks} check(s) passed")
            return 1
        print(
            f"all {self.checks} checks passed"
            + (f", {len(self.unverified)} unverified" if self.unverified else "")
        )
        return 0


def main() -> int:
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=default_root)
    ap.add_argument("--manifest", default=None)
    ap.add_argument(
        "--gitlink-targets",
        choices=("require", "report"),
        default="require",
        help=(
            "require (default): a command target inside a submodule must exist. "
            "report: if it is missing only because the submodule is not checked "
            "out, print it as UNVERIFIED instead of failing. Use report only "
            "where a submodule checkout would need a credential the job must not "
            "hold; the check still has to be run somewhere it can pass."
        ),
    )
    args = ap.parse_args()
    root = os.path.abspath(args.repo_root)
    manifest = args.manifest or os.path.join(root, "benchmark.json")
    return Linter(root, manifest, gitlink_targets=args.gitlink_targets).run()


if __name__ == "__main__":
    sys.exit(main())
