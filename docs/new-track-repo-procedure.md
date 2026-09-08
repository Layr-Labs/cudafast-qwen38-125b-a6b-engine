# New-track repo & branch procedure

How a new model track is stood up across the engine and bench repos. This is the
procedure this repo itself was created by; it applies to every future track family.
Ruled 2026-08-22 (engine fork = new repo; bench = release branch on the shared bench repo).

## 1. Engine fork = a NEW repo per track family

- Create a fresh org repo with a **decoder-neutral** name: `mlxfast-{model}{ver}-{params}-engine`
  (no spec-decoder kind — mtp/dflash/dspark — in repo, track, or branch names; model-facts inside
  code/config are fine).
- **Fresh-seed, do not fork-push.** The org ruleset requires verified commit signatures and the
  source engine's early history contains unsigned commits, so a history-carrying push is rejected.
  Seed one signed commit whose tree is identical to the source engine's `main` tip, and name the
  source commit in the seed message. History stays in the source repo.
  - Precedent: this repo's `main` root `1c50f3f6` = tree-identical seed of
    `mlxfast-qwen-38-27b-mtp-engine @ 31dee355`.
- The repo is created **private**; the visibility flip to internal is org-owner-gated.

## 2. Bench side = a RELEASE BRANCH on the shared bench repo

benchd (the scoring stack) stays in the one shared bench repo. Each track gets a release branch
named exactly the track id, one string serving three roles:

```
{model}{ver}-{params}-{platform}-v{N}  =  bench release branch  =  track_id  =  R2 key prefix
```

- The branch is cut from the bench repo's current `main` at the track's **re-baseline step** — a
  deliberate, reviewed baseline choice, not a float.
- Track-specific measurement work (measure modes, wire-crosscheck fixture, calibration series)
  lands on the release branch.
- Precedents: qwen MLX track → `main`; CUDA track → `qwen3.8-27b-cuda-v1`;
  this track → `qwen3.8-125b-a6b-cuda-v1`.

## 3. Engine ↔ bench binding: the submodule pin

- benchd rides as a **SHA-pinned submodule**: the gitlink IS the pin. `.gitmodules` carries a
  `branch` hint naming the release branch (never a SHA in the comment — it goes stale).
- Verify the pin and its branch containment:

  ```sh
  git ls-tree HEAD benchd
  git -C benchd branch -r --contains $(git rev-parse HEAD:benchd)   # must list the release branch
  ```

  The containment check must use a **remote-tracking** branch (`-r`); local branches inside
  `.git/modules/benchd` prove nothing.
- Gitlink advances are **deliberate two-verdict PRs**, never `submodule update --remote` drift.

## 4. Re-baseline discipline

A seeded engine inherits the **source track's** gitlink, which may be far behind the bench repo's
current `main`. Before any new measurement logic lands:

1. Cut the release branch on the bench repo from current `main`.
2. Advance this repo's gitlink onto that branch tip, and set the `.gitmodules` branch hint,
   **in its own reviewed PR** (the scoring stack is load-bearing; the bump is a pinned identity).
3. Only then build track measurement features on top.

## 5. Goldens for the new track

- Authored **on the track's designated benchmark hardware**, never a development laptop
  (greedy-decode argmax near-ties differ across silicon).
- **A≡B double-generated** — two independent generations, byte-identical asserted **before**
  pinning. Non-identical = STOP; it is also the determinism tripwire for the track's pinned
  runtime configuration.
- Identity = **sha256 + bytes**, never name/path/location. The gates-bound golden pin is the
  oracle-carrying file's hash; the oracle is mandatory on the timed path.
- Uploaded to R2 under `{track_id}/{sha256}.json` (content-addressed, name-free), append-only,
  per-instance authorization, operator-workstation credentials only, GET + sha + bytes
  round-trip verified after upload.
- Upload happens **once the track is stable and ready for testing** — after the engine port and
  measurement stack are proven on-box, before the first scored window.
  `official_scoring_enabled` flips true LAST, in its own PR, after one clean scored window.
