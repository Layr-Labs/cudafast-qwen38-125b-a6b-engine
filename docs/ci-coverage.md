# CI coverage: what the hosted pipeline gates, and what it cannot

`.github/workflows/ci.yml` is the only automated gate in this repository. This
document says what that pipeline proves. More importantly, it says what the
pipeline does not prove, because this package's most interesting tests need
hardware that GitHub does not rent.

This document exists to enforce one rule: **a check that CI cannot run is
named here, never silently skipped.**

## Why the split exists

The scored engine is the `cuda-engine` adapter, which builds and tests without a
GPU, so its unit tests run on hosted CI. Two resources are box-only and decide
whether a test can run on a hosted runner:

- **A CUDA GPU with the ds4 engine built.** GitHub-hosted runners have no GPU.
  A serve is box work that fails closed off the box. The BUILD is different
  from the RUN: `nvcc` needs the toolkit and an aarch64 host, but no device.
  CI therefore does both halves. The `lint` job syntax-checks the engine and
  the shim without CUDA (`tools/ds4/build.sh --cpu-check`), and the
  `cuda-build` job
  compiles the engine for real on a hosted aarch64 runner, inside a
  digest-pinned CUDA devel container, with the toolkit's driver stub in place
  of a driver. That container is **CUDA 13.0** (nvcc 13.0.88), because a stock
  DGX Spark ships CUDA 13.0.x and the goldens are authored on that toolchain.
  CI must compile with the toolkit family the fleet runs: a gate that builds
  with a different one can go green on code the fleet cannot build. The
  engine's RC host is on 13.3.73, and the same tree was measured on both --
  13.0 passes identically and is only slower to compile. Nothing in this pipeline runs the engine against a GPU, and no
  timing claim may ever be made from it.
- **The target checkpoint.** The GGUF target is organizer-staged on the ranked
  box and verified there against its `{sha256, bytes}` pins. It is not in the
  repository, and this pipeline deliberately does not fetch it. Setup's fetch
  and verify path runs in CI against a tiny stand-in served from a `file://`
  base, never against the real checkpoint.

Everything that needs neither resource compiles and runs on hosted hardware,
and that is what CI gates. This covers the adapter's protocol encoding and
decoding, config and artifact contracts, safetensors header parsing, scoring
arithmetic, chat templating, wire fixtures, and request validation.

## CI-covered

| Job | Runner | Check |
| --- | --- | --- |
| `lint` | `ubuntu-latest` | `tools/ci-workflow-egress-scan.sh` — nothing in `.github/` matches a credential-or-egress tripwire pattern (see [Deliberate non-goals](#deliberate-non-goals) for the exact set) |
| `lint` | `ubuntu-latest` | `actionlint` over `.github/workflows/`, with the ranked runner's label declared in `.github/actionlint.yaml` |
| `lint` | `ubuntu-latest` | shell syntax (`bash -n` / `sh -n`) over every tracked `*.sh` |
| `lint` | `ubuntu-latest` | `tools/lint-benchmark-manifest.py --gitlink-targets report` — the Yukon track manifest at rest |
| `lint` | `ubuntu-latest` | `cargo fmt`/`clippy -D warnings` (with and without `--features ds4-engine`)/`test` on `harness/protocol-adapter` — the adapter, against its mock backend and a scripted ds4 session |
| `lint` | `ubuntu-latest` | `tools/test-cuda-engine-staging.sh` — the engine build cache: a restore stages byte-identical content to a build, an incomplete or foreign-workspace entry is refused rather than run, and `setup.sh` refuses a staged engine whose MANIFEST names a different `ds4` commit |
| `lint` | `ubuntu-latest` | `tools/test-setup-onboarding.sh` — `setup.sh` step 2 against a tiny pinned snapshot served from a `file://` base: a fresh machine fills the default directory and verifies it, a rerun on the same machine fetches nothing and reads no shard (the marker answers), a truncated file is fetched again alone, a tampered file and a corrupt source are refused by name, a changed pin invalidates the marker, and the download skip keeps its meaning |
| `lint` | `ubuntu-latest` | `tools/test-local-baseline.sh` — the real `tools/local-baseline.sh`, root proxy and facade with a stub benchd: the defaults and every override reach benchd's argv, an inherited paired-run environment is cleared, and a missing engine, snapshot or golden stops before benchd is spawned |
| `lint` | `ubuntu-latest` | `tools/test-qwen4exp-g1-no-output.sh` — the G1 driver refuses a leg that generated no tokens BY NAME, parsing the engine's counts in both timing-line shapes, so a zero-output regression can no longer read as a pass |
| `lint` | `ubuntu-latest` | `tools/test-participant-engine-edit.sh` — a participant may edit the engine: the real surface gate admits a `ds4/` edit and refuses a `fixtures/` one, benchd's harness hash does not move for an engine edit but does for a harness or `tools/` edit, and an engine edit changes the build-cache key and still builds |
| `lint` | `ubuntu-latest` | a fresh `git clone` of the checkout carries the engine and passes `tools/ds4/build.sh --cpu-check` with no submodule step and no credential |
| `lint` | `ubuntu-latest` | `tools/ds4/build.sh --cpu-check` — the vendored engine and the C shim agree. Runs unconditionally now: the engine is vendored, so there is nothing to fetch and nothing to skip |
| `cuda-build` | `ubuntu-24.04-arm` | `tools/ds4/build.sh` — the real `nvcc` build of the ds4 engine, off-box, in the digest-pinned CUDA devel container (`tools/ds4/ci-container-build.sh`). `libds4qwen.so` links under `--no-undefined`, so a missing engine symbol reds this job instead of failing when benchd spawns the worker. SKIPPED with a `::warning::`, like the `lint` step above, on a runner that cannot read the internal pin |
| `cuda-build` | `ubuntu-24.04-arm` | `cuobjdump --list-elf` — every kernel cubin in `libds4qwen.so` carries `sm_121a` SASS, the arch the ranked box runs |
| `cuda-build` | `ubuntu-24.04-arm` | `cargo test --release --features ds4-engine` — the adapter LINKED against the real engine library, which `lint` can only type-check |
| `cuda-build` | `ubuntu-24.04-arm` | `tools/ds4/build.sh` again with `LIBRARY_PATH` removed — the stub fallback that lets a host without the driver dev symlink link `libds4qwen.so` reports itself and still carries the link |

### This pipeline is advisory

**This pipeline is ADVISORY, and that is the accepted posture** (ruled by David
2026-08-20: the two-session red-team is the de facto merge gate). CI reds the run
on a regression in the checks above. **No status
check is required anywhere in this repository, and a red run blocks nothing
mechanically** — not a merge, not a dispatch:

- **there are no required status checks.** Branch protection is unavailable on
  this repository's plan — the REST endpoint answers `403 Upgrade to GitHub Pro
  or make this repository public` — so no required-status-check rule exists, or
  can be configured, for any job in this file;
- there is no `CODEOWNERS` in this tree, so nothing mechanically requires a
  review either;
- the ranked workflow (`.github/workflows/qwen38-mtp-ranked-benchmark.yml`) is a
  fail-closed stub;
- a **ranked** `workflow_dispatch` does not consult `ci.yml`, so dispatching a
  benchmark does not consult these checks. (This file carries its own
  `workflow_dispatch` trigger, for re-running CI by hand — a different thing.)

None of that is a defect to be worked around. What holds the line is a person
who reads the run, plus the independent red-team passes that gate a merge.
Dispatch-time gating is separate work that lands with the ranked runner: see
`docs/submission-restriction-spec.md` §9 (not ported).

Do not describe any check in this file as "required". Do not add prose implying
that it blocks a merge or a dispatch. If anyone ever configures required
checks, change this section first.

## Box-only: real gates that CI does NOT run

### No in-repository box tests remain

The seed's MLX-runtime box tests (the runtime-worker, cohort, correctness, and
requant/bind suites) were removed with the Apple-Metal runtime, and the entire
Swift package -- with every `Tests/` file and the `MLXFAST_RUN_MLX_RUNTIME_TESTS`
gate scaffolding and its `tools/ci-box-only-inventory.sh` inventory -- went with
the final de-Swift. The scored engine is now the Rust adapter, whose unit tests
run in CI against a mock transport. There is no in-tree test that needs a box; the
box work is the engine build, the checkpoint verification and the measurement, all
listed below.

### Other gates CI does not attempt

| Not run in CI | Why | Where it runs |
| --- | --- | --- |
| ~~The two `benchd/scripts/benchmark.sh` command targets in `benchmark.json`~~ | **No longer applicable.** benchd stopped being a source submodule; every `benchmark.json` command target now lives in this repository (`./setup.sh`, `./tools/fetch-benchd.sh`, `./tools/qwen38-125b-a6b-measure-and-score.sh`), so CI verifies all three and the linter downgrades nothing | CI, fully verified |
| ~~The ds4 CUDA build~~ | **No longer applicable.** The build needs the CUDA toolkit and an aarch64 host, not a device, so the `cuda-build` job does it on a hosted runner | CI, and again on the box via `./setup.sh` |
| The GGUF checkpoint verify | needs the organizer-staged checkpoint CI cannot provide | box, via `./setup.sh` |
| Execution of any CUDA kernel the `cuda-build` job compiles | the container has no device, and its CUDA driver library is the toolkit's stub, which cannot run. `cuda-build` proves the code compiles, links and targets the right arch; it proves nothing about numerics or speed | box |
| The scored engine driven against the real ds4 engine | the adapter build and its revert-proof staging are CI-covered (`tools/test-cuda-engine-staging.sh`); the real drive needs the box | box, `tools/ds4/mtp-exactness-gate.py` before a benchd pass |
| Correctness gates, token fidelity, the timed paired measurement, scoring | measurement authority is benchd's, on the ranked M5 box, behind a thermal gate | the ranked pipeline; nothing in this repository times or scores anything |
| Hidden-golden fidelity | hidden material, R2-provisioned, sha256+bytes pinned | ranked box only |

## Deliberate non-goals

- **No secrets.** Not a token, not a deploy key, not an R2 credential.
- **No artifact upload from CI.** This repository can receive organizer-material
  fixtures, and the runner must not be a way out for them. The one exception is
  the ranked pipeline's score file, which is the product Yukon reads back:
  `.github/workflows/benchmark.yml` uploads `score.json` as
  `benchmark-results-<run_id>`. The scan checks that upload rather than
  exempting it. **Exactly one** upload reference may exist in that file. That
  one step's own block must carry a 40-hex-pinned action, `path: score.json`,
  and `if-no-files-found: error`. A second upload step fails the scan however
  well-formed it is. The three property checks are evaluated inside the checked
  step, so another step cannot vouch for it.
- **Both are tripwired, not proven.** `tools/ci-workflow-egress-scan.sh` scans
  all of `.github/` — workflows and any composite action beside them — and
  fails the run on a named set of patterns: a `secrets` reference
  (`secrets.NAME`, `secrets['NAME']`, `toJSON(secrets)`), `secrets: inherit`,
  any `write` permission (blanket or single-scope, `id-token: write`
  included), and an artifact upload by action or by expression anywhere other
  than the checked ranked-score upload above. That set is
  the whole of what it detects. It cannot see a `run:` step that curls, a
  third-party action that uploads on its own, or an `actions/cache` entry used
  as a side channel. Those remain review's job. The scan is the floor that
  stops the common accidental regressions landing silently. It is not a proof
  that nothing leaves the runner.
- **No ranked measurement.** `.github/workflows/qwen38-mtp-ranked-benchmark.yml`
  is a fail-closed stub and stays one. CI does not dispatch it, and a
  `workflow_dispatch` of it exits 1 by design.
- **No submodules at all.** The `ds4` engine is vendored into this repository, so every engine step runs from the checkout with no credential.
