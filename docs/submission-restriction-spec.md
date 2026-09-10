# Submission-restriction enforcement: extracted spec and parity notes

Issue #16. This document is the spec-extraction half of that issue: what the
ORIGINAL challenger implementation actually enforces about an untrusted
submission, rule by rule, with a citation for each; what this repository
re-implements; where the re-implementation deliberately differs; and which
questions in the original need an operator ruling before anything dispatches.

> **NOTE — the Swift byte-budget enforcer (R1) was REMOVED in the final
> de-Swift (Stage 5a).** The upstream `EditableSurfaceByteBudget.swift` and the
> `@bfab0de:` citations below still describe the challenger reference, which is
> intact and read-only. This repository's own re-implementation of R1 —
> `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift` and its
> `mlxfast-swift` launch path — is gone with the rest of the Apple-Metal/Swift
> seed package: it was a parity twin that was never wired into a live launch
> path here (nothing in this repository called it at worker spawn), so removing
> it changed nothing on the scored/enforced surface. **The one live enforcer of
> the byte caps is the pre-dispatch shell static review**
> (`.github/scripts/submission-static-review-checks.sh`, §5), which runs in the
> ranked pipeline's hosted surface-check job before any dispatch reaches the
> box; `tools/lint-benchmark-manifest.py` check 3b holds its fallback constants
> to the manifest. Read
> every "launch-time byte budget / EditableSurfaceByteBudget / R1" mention below
> as the upstream reference and this repository's removed twin, not as a gate
> that runs here today.

## 0. Provenance of every citation

The reference is the challenger repository, read-only:

```text
Layr-Labs/qwen-3.8-mtp-challenge   main @ bfab0de58d43453e506523707e1720a3485570f4
```

fetched 2026-08-20 into this repository as the `upstream` remote. Citations are
written `<path>@bfab0de:<line>` and every one is reproducible with:

```bash
git -C <engine-checkout> fetch upstream
git -C <engine-checkout> show upstream/main:<path> | sed -n '<line>p'
```

The offline ranked-box mirror in `b4-ranked-box-mirror/` carries an older
checkout (`1af12bc`, 2026-08-14) of the same tree. It was consulted only to
confirm the files exist there too; nothing in this document is cited from it,
because `upstream/main` is both newer and independently verifiable.

The files this document extracts rule by rule — the gates that decide what
a submission may CONTAIN and how much of it — are:

| File | Role |
|---|---|
| `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift` | launch-time byte budget (upstream; this repository carries no Swift package) |
| `.github/scripts/run-submission-static-review.sh` | pre-dispatch deterministic caps + LLM bypass judge |

`.github/scripts/hardened-git.sh` supports the static review. The in-repository
overlay and modifiable-surface gate that this document once extracted (R3, R4)
were retired on 2026-09-09: neither had a caller on any scoring path. Yukon's
service overlays the archive, and benchd enforces the surface at scoring (§1).

**These four are not the whole submission-restriction surface upstream.** At
least three further gates are load-bearing against a hostile submission and are
NOT extracted here; §9 records each as not-ported, with what it would cost to
be wrong about it:

| File | Role | Why it matters here |
|---|---|---|
| `.github/scripts/verify-trusted-source-scope.sh` | trusted-source-scope re-verification immediately before the trusted build | carries the GENERAL form of this repository's A1: `verify_contract_does_not_expose_trusted_scope@bfab0de:117-139` refuses any `editablePaths` entry overlapping the trusted scope, read from the TRUSTED ref. A1 protects one pair of paths (`benchd`, `.gitmodules`) by construction; this protects the whole trusted scope. The run-time half — trusted-ref contract read plus byte-identity re-verification of the scope before the trusted build — is deferred to runner provisioning. A rest-state overlap check briefly stood in as linter check 3c; it was removed 2026-08-30 (David: no unnecessary guards). See §9. |
| `.github/scripts/pin-trusted-harness.sh` | content pins over the built trusted and worker artifacts | binds every scored phase to the binaries built from the reviewed source, so a submitted transform cannot rewrite the worker after the build |
| `.github/scripts/deny-private-artifacts.sh` | artifact/cache upload refusal | bounds exfiltration of hidden prompts, goldens and shards through the artifact channel |

The earlier revision of this section claimed the four-file table was the whole
surface. It was false, and the fail-closed reading is the one written above:
treat the extracted four as a floor, not a boundary. Nothing in §§2-5 depends on
the difference — those sections cite the four files directly — but §1's "four
independent gates" is likewise a statement about what this repository
re-implements, not a census of upstream.

## 1. Layer map

An untrusted submission passes three gates. One runs in this repository; the
other two are outside it. Each is fail-closed on its own; none is load-bearing
alone.

```text
archive ──► Yukon's overlay                REPLACE only editablePaths, read
            (Yukon's service)              from the trusted contract; an absent
                                           OPTIONAL path keeps the trusted copy
   │
   ├──────► static-review checks           per-file / total / growth byte caps,
            (this repository, hosted)      path validation, exempt handling
                                           ── then the LLM bypass judge
   │
   └──────► benchd, at scoring             refuses a candidate that diverges
            (the trust boundary)           from its baseline outside the surface
```

The upstream launch-time byte budget (`EditableSurfaceByteBudget`) does not run
here: this repository carries no Swift package. benchd applies the byte budget
at scoring.

The double enforcement of the byte caps is explicit in the original:
`EditableSurfaceByteBudget.swift@bfab0de:4-13` states that the judge applies the
policy but "these mechanical caps are the launch-time backstop that binds every
ranked worker launch path, including dispatches that never pass through the
review step."

## 2. R1 — launch-time byte budget

Source: `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift@bfab0de:1-157`,
called from `Sources/MLXFastCLI/main.swift@bfab0de:2286` inside the runtime-worker
spawn preflight, implemented at `main.swift@bfab0de:2361-2397`.

| # | Rule | Original |
|---|---|---|
| R1.1 | Total editable surface ≤ 3,000,000 bytes | `bfab0de:18` |
| R1.2 | Any single editable file ≤ 524,288 bytes | `bfab0de:19` |
| R1.3 | Contract is `benchmark.json` next to the cwd | `bfab0de:20`, `main.swift@bfab0de:2362-2364` |
| R1.4 | Absent contract = `.skipped`; official runs treat that as fatal, local runs proceed | `bfab0de:57-59`, `main.swift@bfab0de:2380-2386` |
| R1.5 | Contract that does not parse, or carries no non-empty `editablePaths`, = `.exceeded` (fail closed) | `bfab0de:60-68` |
| R1.6 | Walk is rooted at the contract's own directory | `bfab0de:75` |
| R1.7 | Upstream: only regular files count; symlinks and other non-regular entries are SKIPPED, not rejected. DIVERGED here — a symlink at or under an editable path fails closed (D8) | `bfab0de:37-39`, `bfab0de:118-124`, `bfab0de:138-144` |
| R1.8 | A missing editable path is skipped silently; DIVERGED — a surface where EVERY path is absent (walks to zero files) fails closed (D8) | `bfab0de:115-117` |
| R1.9 | A directory that cannot be enumerated = `.exceeded` | `bfab0de:133-137` |
| R1.10 | `editableSurfaceByteBudget.exemptPaths` are held out of the CODE budget and charged to their own cap. **This track declares no `exemptPaths` since the requant-only ruling (2026-08-26), so the exempt arm is unreachable here.** The mechanism is retained in the enforcers and still tested against a synthetic contract | `bfab0de:41-50`, `bfab0de:70-73`, `bfab0de:98-109` |
| R1.11 | Exempt AGGREGATE cap defaults to 512,000,000 (David BYO-512 ruling 2026-08-26). It is still DECLARED and still drift-checked against both enforcers' fallbacks, and it cannot bind while `exemptPaths` is absent | `bfab0de:22-25` |
| R1.11a | Exempt PER-FILE cap defaults to 100,000,000 and is checked before the aggregate, so an oversize submitted blob names itself. Same status as R1.11: declared, drift-checked, unreachable while `exemptPaths` is absent. This cap is what refused the organizer's own monolithic 236 MB staged MTP head on the box before the head directories left `editablePaths` | added 2026-08-26 |
| R1.12 | The per-file cap is checked BEFORE the running total, so an oversized file names itself | `bfab0de:81-86` |
| R1.13 | The total is checked after each file, so the message says "at least N bytes" | `bfab0de:89-94` |
| R1.14 | Official run: `.exceeded` refuses the worker spawn; local run: warn on stderr and continue | `main.swift@bfab0de:2387-2396` |
| R1.15 | Env overrides `MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES` / `_MAX_FILE_BYTES`, same knobs as the review script — **NOT ported; retired by ruling, see §8 Q9** | `main.swift@bfab0de:2365-2372` |

Why the exemption existed, in the original's own words
(`bfab0de:41-50`): head weights joined the competitive surface on 2026-08-14,
they are not source, a 3 MB source budget would make in-branch head delivery
unusable, and what bounds an exempt path instead is the digest + byte-count
verification of `mtp-head.manifest.json` done before the sandbox opens. The
lookup-table defence is untouched for every other path.

**THE EXEMPTION IS RETIRED ON THIS TRACK (David requant-only ruling,
2026-08-26).** In-branch head delivery is retired with it: no head weights
directory is an `editablePaths` entry, the head declaration accepts
`"source": "pinned"` only, and a submission carries no head weights. On this
track the head is embedded in the pinned target checkpoint, so nothing stages a
head artifact at all. There is therefore nothing left to exempt, and
`exemptPaths` is absent from this repository's contract. The paragraph above is
kept as the record of what the exemption was for.

**"Digest + byte-count" is the original's claim, not this repository's bound —
RATIFIED size-only (engine-alteration-blocks audit, 2026-08-21; DECIDE-2 Q-B).**
Here the digest half of that sentence is deliberately not a gate. That remains
true after the requant-only ruling, and it is now the sharpest open limit on the
track: `max_bytes` (2 GiB) is enforced fail-closed, and a declared `sha256` is
parsed and carried when present but neither required nor verified against the
head bytes
(`Sources/MLXFastTrustedHarness/Gemma4MTPHeadDeclaration.swift`, the
"DECIDE-2 Q-B" comment; the sealed record's `head_provenance` carries the
harness's own recomputed tree digest, never the declared value — see
`docs/qwen38-125b-a6b-port-notes.md` section 4.2). The compensating control
is not content identity but isolation: the measured leg runs the head inside
the box-enforced bench-560 sandbox (the PF-isolated bench uid of mlxfast-bench,
specified in the bench repository's own measure-job contract), and both legs
load the head out of the one pin-verified target checkpoint — the serial
denominator always runs the organizer-pinned head.
A tighter digest/exact-byte gate was considered and rejected as contradicting
that ruling; see D8 in §7 and the #20 items in §8, which record the boundary
between the fail-closed fixes (a–d) and this size-only ruling (e).

**Re-implementation:** `Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift`
in this repository. R1.1–R1.14 hold. Divergences D1–D4 in §7.

## 3. R2 — static review, deterministic half

Source: `.github/scripts/run-submission-static-review.sh@bfab0de:1-412`. Lines
414-698 of that file are the LLM bypass judge and are NOT ported (§9).

| # | Rule | Original |
|---|---|---|
| R2.1 | Caps: total 3,000,000 / per-file 524,288 / growth 262,144, each overridable by env, each required to be a positive integer | `bfab0de:36-38`, `bfab0de:98-109` |
| R2.2 | The growth cap exists because the total cap alone lets a small diff hide a large table | `bfab0de:348-353` |
| R2.3 | Contract must exist and be non-empty | `bfab0de:94-97` |
| R2.4 | Track id set-but-empty is fatal; an unknown track id is fatal; reviewing under another track's policy inverts every verdict | `bfab0de:40-88` |
| R2.5 | Editable path validation: non-empty, not absolute, no `\`, no leading `:` (git pathspec magic), no `/../` or `/./` component | `bfab0de:136-149` |
| R2.6 | Exempt paths are validated as paths too — they also end up in a git pathspec | `bfab0de:277-281` |
| R2.7 | In diff mode the allowlist AND the exemptions are read from the BASE commit, never from the submission work tree | `bfab0de:249-260` |
| R2.8 | `MLXFAST_SUBMISSION_REVIEW_BASE_SHA` set-but-empty is fatal — never degrade to whole-surface review | `bfab0de:221-227` |
| R2.9 | Base and head must resolve to commits, and the work tree must BE the review head, or the collected bytes would not match the reviewed diff | `bfab0de:235-248` |
| R2.10 | An empty allowlist is fatal, never an accidental clean pass (a jq failure inside a process substitution is invisible to `set -e`) | `bfab0de:270-275` |
| R2.11 | Exempt paths keep editable status but never enter the payload, the diff, or the growth arithmetic | `bfab0de:283-293` |
| R2.12 | A surface where every entry is exempt is fatal | `bfab0de:294-297` |
| R2.13 | Zero changed paths = clean pass, recorded as such | `bfab0de:342-346` |
| R2.14 | Growth is computed from COMMITTED blob sizes on both sides so the work tree cannot skew it | `bfab0de:354-369` |
| R2.15 | Editable files deleted vs the base are reported as the stale-clone signature and the hint is appended to any later oversize refusal | `bfab0de:319-340` |
| R2.16 | Every changed non-deleted path must be a regular file (not a symlink) in the checkout | `bfab0de:371-378` |
| R2.17 | The unified diff counts against the same total cap | `bfab0de:380-395` |
| R2.18 | Whole-surface mode (no base): every reviewed path must EXIST, and selecting zero files is fatal. Upstream selects this mode by the ABSENCE of a base sha; here it is an explicit opt-in — see D7. DIVERGED — whole-surface now also rejects a symlink or non-regular entry under a reviewed path, the check R2.16 carried only in diff mode (D8) | `bfab0de:396-412` |
| R2.19 | Every git read goes through `hardened-git.sh`, resolved next to the script (the trusted copy), never one inside the submission worktree | `bfab0de:5-14` |

`hardened-git.sh@bfab0de:1-46`: `env -i` with only `PATH` restored,
`GIT_CONFIG_GLOBAL`/`_SYSTEM` to `/dev/null`, `GIT_CONFIG_NOSYSTEM=1`,
`HOME=/var/empty`, and `-c core.fsmonitor=false -c core.hooksPath=/dev/null -c
core.pager=cat -c protocol.ext.allow=never -c safe.directory='*'`. The threat is
a planted repo-local `.git/config` running arbitrary programs as the invoking
uid.

**Re-implementation:** `.github/scripts/submission-static-review-checks.sh` and
`.github/scripts/hardened-git.sh`. R2.1–R2.19 hold except as noted in §7 (D5,
D6).

## 4. R3 — overlay, REPLACE semantics

Retired 2026-09-09. The in-repository re-implementation
(`.github/scripts/overlay-editable-paths.sh`) had no caller on any scoring
path: Yukon's service performs the overlay of a submission archive. The rules
below describe the original for the record; R3.6 (an absent optional path keeps
the trusted copy) is the one the participant contract still states.

Source: `overlay@bfab0de:1-186` (the original).

| # | Rule | Original |
|---|---|---|
| R3.1 | `SUBMISSION_WORKTREE` required; missing worktree, missing trusted contract, or missing `jq` are all fatal | `bfab0de:5-20` |
| R3.2 | Editable path validation: non-empty, not absolute, no `\`, no `/../` or `/./` | `bfab0de:22-34` |
| R3.3 | The allowlist and the optional set come from the TRUSTED contract in the cwd, never from the submission worktree | `bfab0de:7`, `bfab0de:39`, `bfab0de:158-160`, `bfab0de:184` |
| R3.4 | REPLACE, not merge: `rm -rf` the target, then copy the submission's copy wholesale | `bfab0de:173-180` |
| R3.5 | A missing REQUIRED editable path is a loud refusal (it means a stale clone) | `bfab0de:165-166` |
| R3.6 | A missing OPTIONAL editable path is SKIPPED and the trusted copy is kept — absence means "use the organizer-pinned head", never "delete it" | `bfab0de:145-164` |
| R3.7 | Optional matching is exact string equality against `optionalEditablePaths` | `bfab0de:41-47` |
| R3.8 | Any symlink anywhere under a submitted editable path is a refusal, checked BEFORE the copy | `bfab0de:168-171` |
| R3.9 | After the copy the target tree must contain no symlink, no non-regular entry, no hardlinked file, no setuid/setgid file | `bfab0de:49-67`, `bfab0de:181` |
| R3.10 | Stale-editable-file detection is REPORT-ONLY and silently skipped without `TRUSTED_MAIN_SHA` or a resolvable merge base | `bfab0de:69-137` |

R3.6 is the rule the whole optional mechanism exists for, and the original
explains why (`bfab0de:148-160`): archives have REPLACE semantics, the head
declaration is a small opt-in file most archives will not carry, the contract
says an absent `mtp-head.manifest.json` means the organizer-pinned head, and
failing the overlay would turn the default case into a refusal. The optional set
is read from the trusted contract precisely so "a submission cannot make its own
missing source files optional."

**Re-implementation:** none. Retired; see above.

## 5. R4 — modifiable-surface gate

Retired 2026-09-09. The in-repository re-implementation
(`.github/scripts/enforce-modifiable-surface.sh`) had no caller on any scoring
path. What enforces the surface is benchd at scoring: a candidate that diverges
from its baseline outside `editablePaths` is refused. The rules below describe
the original for the record.

Source: `enforce@bfab0de:1-77` (the original).

| # | Rule | Original |
|---|---|---|
| R4.1 | `BASE_SHA` and `HEAD_SHA` required | `bfab0de:7-8` |
| R4.2 | The allowlist is read from the BASE commit so a PR cannot grant itself access | `bfab0de:2-3`, `bfab0de:34-47` |
| R4.3 | `CONTRACT_PATH` is deliberately the SAME variable the overlay and the review read, so the three gates cannot disagree about which contract binds | `bfab0de:10-16` |
| R4.4 | No fallback to `benchmark.json` on an unreadable contract — judging a submission against another track's surface would reject it naming a contract it never used | `bfab0de:38-43` |
| R4.5 | A contract with no, or an empty, `editablePaths` is fatal | `bfab0de:48-57` |
| R4.6 | A changed file matches the surface by exact equality OR by allowed-directory prefix | `bfab0de:61-76` |
| R4.7 | Every git read goes through `hardened-git.sh` from the trusted checkout | `bfab0de:18-27` |

**Re-implementation:** none. Retired; see above.

## 6. R5 — Seatbelt profile generation (KNOWN LEDGER ITEM, not resolved here)

Issue #16 lists the Seatbelt generation divergence as an already-tracked ledger
item, so this section records what the divergence IS and stops there.

Two generators produce the runtime-worker sandbox profile:

* bash — `benchmark.sh@bfab0de:1266-1307` (`write_runtime_worker_sandbox_profile`)
* Swift — `Sources/MLXFastCLI/main.swift@bfab0de:2605-2652` (`writeRuntimeWorkerSandboxProfile`)

They agree on `(allow default)`, `(deny network*)`, `(deny process-fork)`,
`(deny process-exec*)` + a literal allow for the worker binary, `(deny
file-write*)` + `/dev/null`, and the golden/private-dir read denials.

They differ in one security-relevant rule. The Swift generator emits three
`mach-lookup` denials for the mDNS resolver
(`main.swift@bfab0de:2643-2645`) with the reasoning at
`main.swift@bfab0de:2626-2635`: `(deny network*)` blocks the worker's own
sockets, but `getaddrinfo(3)` resolves via IPC to `mDNSResponder`, which
egresses from ITS uid, so a uid/socket-scoped block never sees the DNS query and
submitted code could exfiltrate over DNS. The bash generator
(`benchmark.sh@bfab0de:1287-1304`) emits no `mach-lookup` rule at all.

Consequence: any dispatch path whose profile comes from the bash generator has a
DNS-shaped exfiltration channel the Swift-generated profile closes. This
repository ships neither generator on a dispatch path today; the divergence is
recorded so whichever generator the provisioned runner adopts is the hardened
one.

## 7. Divergences in this repository's re-implementation

Each is deliberate.

**D1 — the caps come from the manifest, not from constants.**
`benchmark.json`'s `editableSurfaceByteBudget` now carries machine-readable
`maxTotalBytes` / `maxFileBytes` / `maxGrowthBytes` alongside the existing
`exemptPathMaxBytes`. Both enforcers read them. The compiled-in constants and
the shell defaults survive only as the fallback for a contract that declares
none, and three drift tests assert they still equal the manifest. Upstream, the
numbers lived in a Swift constant (`bfab0de:18-19`) and a shell default
(`review@bfab0de:36-38`) kept in sync by a comment
(`review@bfab0de:32-35`); the manifest stated them in prose only. Issue #16
directs the manifest to be the single source.

**D2 — a present-but-malformed cap fails closed, in BOTH enforcers.** Upstream
coerced a malformed `exemptPathMaxBytes` to the default (`bfab0de:72-73`), so a
typo silently replaced the operator's intent with 2 GiB. Here the contract is
decoded with `JSONDecoder`, which refuses `true`, `"2000000"` and `3.5` for an
`Int`, and a non-positive value is rejected explicitly. Two things make the rule
hold rather than merely be stated:

* The Swift decoder is hand-written for the cap keys. The synthesized
  `decodeIfPresent` reports an explicit JSON `null` as ABSENT, which would have
  coerced `"maxTotalBytes": null` to the fallback; only a genuinely missing key
  takes the fallback now, one level up (`editableSurfaceByteBudget: null`)
  included.
* The shell enforcer type-gates on the JSON type instead of `jq -r ... // empty`.
  `// empty` read `false` and `null` as absent (silent fall to the constant) and
  `-r` stringified `"2000000"` into a value that passed the integer regex — a
  submission-supplied *string* silently widened the cap it was measured against.
  The two enforcers now agree on `"2000000"`, `"99999999"`, `false`, `null`,
  `3.5` and `5e5`.

**D3 — `maxFileBytes > maxTotalBytes` is rejected.** A per-file cap that can
never bind is a configuration error that reads as a working defence. Upstream
did not check this.

**D4 — the unused `import MLXFastCore` is dropped.** Upstream imported it
without using it (`bfab0de:2`). Removing it lets the E2E suite compile the real
enforcer source directly with `swiftc` in about a second, so the suite tests the
shipped implementation instead of a copy of its rules. `swift build` compiles
the same file into the trusted-harness target unchanged.

**D5 — the static-review track default is the contract's, not `serial`.**
Upstream defaults `TRACK_ID` to `serial` when the variable is unset
(`review@bfab0de:51`). In this repository that default would review a legal
native-MTP submission under the retired serial track's total speculation ban,
where every legal submission is a critical finding. The re-implementation
defaults to the manifest's `staticReviewTrackId` and rejects any explicitly
requested track that is not the one the contract declares.

**D6 — the LLM judge half is not ported.** See §9.

**D7 — whole-surface mode is an explicit opt-in, not the fallback.** Upstream
selects the review mode by the ABSENCE of `MLXFAST_SUBMISSION_REVIEW_BASE_SHA`
(`review@bfab0de:221-227` makes set-but-empty fatal, but unset silently selects
whole-surface). Whole-surface mode reads the contract from the WORK TREE, which
is only safe because it is meant to run in the trusted checkout after the
overlay — a precondition the script cannot verify. As a fallback it put the
entire self-widening primitive one missing environment variable away: an
attacker checkout carrying a contract with `maxTotalBytes: 999999999` and no
base sha reviewed itself against its own caps and exited 0. Here the mode must
be named:

```text
BASE_SHA set-but-empty                    -> fatal (base computation failed)
BASE_SHA non-empty, MODE unset or "diff"  -> diff mode
BASE_SHA non-empty, MODE anything else    -> fatal (contradictory request)
BASE_SHA unset, MODE "whole-surface"      -> whole-surface mode
BASE_SHA unset, MODE anything else        -> fatal (no mode selected)
```

A caller that forgets to compute a base gets a refusal instead of the
permissive mode. Diff-mode selection is unchanged.

**A1 (addition) — the gitlink is untouchable by construction.**
`tools/lint-benchmark-manifest.py` refuses any editable entry equal to, or
inside, a trusted-scope path or a gitlink. An editable entry over a gitlink
would let a submission repoint its own scorer. The original had no submodule to
protect.

The guard is CASE-FOLDED, and separately checks filesystem identity, so an
entry spelled in another case names the same real path and is refused. ASCII
folding normalises the spellings it reaches; `os.path.samefile` over every
prefix of the entry catches the ones it does not. The retired overlay and
surface gate carried the same guard; the linter is where it lives now.

**A2 (addition) — one path-validation rule, not two.** Upstream's overlay
validator (`overlay@bfab0de:22-34`) and its static-review validator
(`review@bfab0de:136-149`) implement the same concept with different rules: only
the latter rejects a leading `:`. This repository applies the stricter rule in
the static review and the linter. See Q1 below.

**D8 — the fail-closed hardening of issue #20 (a–d).** Four upstream behaviours
that read as clean passes are refusals here (§8 Q2/Q3/Q4/Q8 carry the
detail and the ruling): a non-regular entry — a symlink, or a FIFO/socket/device
(F1) — at or under an editable path is rejected by BOTH deterministic gates
rather than skipped, symmetrically at root and in-directory (Q2, diverges from
R1.7 and extends R2.16's diff-mode check to whole-surface R2.18); an all-absent editable surface
fails the byte budget closed rather than returning a zero-byte `.verified` (Q3,
diverges from R1.8); the exempt-cap overflow message names the aggregate, not a
single path (Q4); and the overlay fails closed on an empty or unparseable
allowlist rather than printing a successful-overlay trailer after moving nothing
(Q8). None widens the surface — every one turns a silent pass into a refusal.
The exempt-channel bound (issue #20 e) is NOT part of D8: it is a ruling item,
not a fail-closed fix. It is also moot on this track since the requant-only
ruling — nothing rides in a submission that the exempt channel could bound. A
digest gate on the STAGED head is a separate, still-open question, and it is
recorded as such in `docs/participant-contract.md` section 4.3: nothing binds
staged head bytes to the organizer's pinned digests at run time.

## 8. Ambiguities in the original that want an operator ruling

None of these block the suite; all of them shape what "parity" should mean.

**Q1 — two different definitions of a legal editable path.** `overlay@bfab0de:22-34`
omits the leading-`:` (git pathspec magic) test that `review@bfab0de:136-149`
carries. Is the laxer overlay rule intentional (the overlay never builds a
pathspec) or drift? We unified on the stricter rule.

**Q2 — the byte budget SKIPS symlinks rather than rejecting them. RULED (engine
issue #20 a): reject, in BOTH deterministic gates.** `bfab0de:37-39` justifies
skipping by saying the overlay and surface enforcement reject them separately.
That is true for a dispatch that overlays. On a path that does not — an
operator-populated checkout, a local official run — a symlink farm is uncounted
by the only gate that runs; whole-surface static review (R2.18) was blind the
same way, since `find -type f` skips a link. Both gates now fail closed on a
non-regular entry — a symlink, or (F1 gate-symmetry) a FIFO/socket/device — both
AT a root editable path and UNDER one:
`Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift` (a divergence
from R1.7 — see D8) and `.github/scripts/submission-static-review-checks.sh`
whole-surface (the check R2.16 already carried in diff mode). The three gates
are symmetric — the budget's root branch, the budget's in-directory branch, and
the shell whole-surface gate all refuse a non-regular file at root, so none can
skip what another rejects.

**Q3 — a missing editable path is treated differently by two gates on the same
surface. RULED (engine issue #20 b): the budget fails closed too.** The byte
budget skipped it (`bfab0de:115-117`) and returned `.verified` totalBytes=0
fileCount=0 when EVERY editable path was absent — absence read as a clean pass;
whole-surface static review calls it fatal (`review@bfab0de:399-402`). The two
now agree: the budget refuses a surface that walked to zero files, mirroring the
shell gate's `file_count == 0` guard.

**Q4 — the exempt cap is an aggregate charged with a per-path message. RULED
(engine issue #20 c): name the aggregate.** `exemptBytes` accumulates across ALL
exempt paths (`bfab0de:100`) but the refusal named only the path being walked
when the cap tripped (`bfab0de:104-106`) — exact with today's single exempt
path, misattributing with two. The message now names the aggregate and the
number of exempt paths, not a single path.

**Q5 — the growth cap has no launch-time counterpart.** `MAX_GROWTH_BYTES`
exists only in the static review, because the launch has no base commit. A
dispatch path that skips review therefore has no growth bound, only the total
and per-file caps — which is exactly the gap R2.2 says the growth cap exists to
close. Accepted risk, or does the launch path need a recorded base?

**Q6 — the setuid check runs on the copy, after the copy stripped it.**
`validate_overlay_tree` (`bfab0de:63-66`) inspects the TARGET, but the
unprivileged `tar`/`cp` at `bfab0de:173-180` already drops setuid/setgid on
extraction. The check can only fire under root, or a `tar` invoked with `-p`.
The SOURCE tree is never checked for setuid. The guarantee that actually holds
is "the bit never lands", and the suite asserts that rather than the refusal.

**Q7 — the byte caps' authority.** Upstream stated 3 MB / 512 KiB in prose in
the manifest, as a Swift constant, and as a shell default. If they ever
disagreed, nothing said which won. D1 makes the manifest win here; that wants
ratifying before it is treated as the contract.

**Q8 — the overlay reports success after overlaying NOTHING. RULED (engine issue
#20 d): fail closed on an empty or unparseable allowlist.** `overlay@bfab0de`
read `.editablePaths[]` in the loop header, so a trusted contract with
`editablePaths` empty, absent, or unparseable produced zero iterations and the
`trusted harness retained; ... overlaid` trailer still printed at exit 0. The
direction is SAFE (the trusted tree is kept, nothing hostile lands) but the
false success lets a pipeline measure the trusted tree and attribute the score
to a submission that overlaid nothing. The allowlist is now read up front and an
empty or unparseable contract fails closed before the trailer.

**Q9 — may an environment variable re-state a run's caps at all? RULED (David
2026-08-20), engine issue #20, review finding B6 / R1.15: align shell to Swift,
`manifest > constant`.** The shell enforcer used to resolve each cap as
`environment > manifest > constant`
(`.github/scripts/submission-static-review-checks.sh`, `resolve_cap`) while the
Swift enforcer had no environment path at all, so D1's "the manifest is the
single source" held for one enforcer and not the other.

`MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES`, `_MAX_FILE_BYTES` and
`_MAX_GROWTH_BYTES` are now **removed** from the shell gate — not deprecated,
not outranked. Both enforcers resolve `manifest > constant` and D1 holds
everywhere. The lever the overrides gave a trusted operator (re-state one run's
caps without editing the contract) is deliberately given up: a cap that can be
re-stated out of band is not a single source, and the reviewed run could differ
from the contract with nothing in the record saying so. Neither direction was
ever a submission-reachable bypass — the variables were trusted-operator inputs
and a submission has no way to set them — so this closed a doctrine divergence,
not a hole.

The shell gate reads none of the three variables, and both enforcers resolve the
caps from the manifest only. Reintroducing an environment path here means
reopening #20.

## 9. Not ported

* **The LLM bypass judge** — `review@bfab0de:414-698`: the track-selected
  controlling decode rules, the system prompt, the Anthropic call, retry
  handling, and the verdict/severity reconciliation at `bfab0de:678-698`. It is
  policy, it needs a credential and a network, and it cannot run pre-dispatch on
  an offline box. The deterministic half is what can be a test and it fails
  closed on its own. When the judge is added, `bfab0de:684-694` is the rule that
  matters most: a schema-valid but self-contradictory verdict —
  `passed=true` with `severity` `high` or `critical`, the shape a partially
  successful prompt injection in the reviewed code produces — is forced to
  FAILED regardless of the reported `passed` value.
* **The Seatbelt generators** — §6, a known ledger item.
* **`verify-trusted-source-scope.sh`** — the fifth load-bearing gate named in
  §0. Two halves: a byte-identity re-verification of every trusted-harness
  source input against the trusted git content immediately before the trusted
  build, and `verify_contract_does_not_expose_trusted_scope@bfab0de:117-139`,
  a standing tripwire that refuses any `editablePaths` entry overlapping the
  trusted scope — read from the TRUSTED ref so an overlaid contract cannot
  steer it. Addition A1 here is the same idea narrowed to `benchd` and
  `.gitmodules`.

  A rest-state overlap check briefly stood in as linter check 3c (it refused
  any `editablePaths` / `optionalEditablePaths` /
  `editableSurfaceByteBudget.exemptPaths` entry overlapping the trusted scope,
  by the same overlap relation upstream uses). It was **removed** 2026-08-30
  under David's no-unnecessary-guards ruling, together with its tests. Addition
  A1 (the gitlink narrowing — `benchd`, `.gitmodules`, `benchd.pin`,
  `benchd-bin` — linter check 3) stays and still runs in CI.

  Both halves of the GENERAL form are therefore absent here: the run-time half
  (reading the contract from the TRUSTED ref rather than the tree, plus
  byte-identity re-verification of every trusted-harness source input before
  the trusted build) is deferred to runner provisioning (#36); the rest-state
  overlap half was removed by ruling. The trusted engine surface itself is now
  participant-EDITABLE on this track (the adapter `harness/`), so a general
  trusted-source-scope tripwire no longer matches the contract's intent.
* **`pin-trusted-harness.sh`** — content pins over the built trusted CLI and
  the participant worker, captured on either side of the participant build so
  build-time code execution cannot alter the trusted binary between build and
  pin. Needs built artifacts; runner provisioning.
* **`deny-private-artifacts.sh`** — refuses artifact and cache uploads that
  could leak hidden prompts, golden tokens, model shards, symlinks, or
  unexpectedly large files. Needs an artifact channel; runner provisioning.
* **The workflow that sequences these gates** — runner provisioning, #36
  cross-repo.
* **The launch-time CALL SITE** — `main.swift@bfab0de:2286`. The enforcer class
  is restored and tested; wiring it into the worker-spawn preflight lands with
  runner provisioning.
