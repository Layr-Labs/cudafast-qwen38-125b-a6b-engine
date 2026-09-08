# tools/box-runner/

Scripts that make a box serve GitHub Actions jobs for this repository's
ranked benchmark: mint a just-in-time (JIT), single-use runner registration,
run exactly one job with it, verify the job's whole process tree is gone,
repeat. This directory owns the **runner side** only. The workflow file that
dispatches jobs against it is `.github/workflows/benchmark.yml`
(`runs-on: [self-hosted, macOS, qwen3.8-125b-a6b-cuda-v1]`); the coupling between
the two sides is exactly the `RUNNER_LABELS` value described below, and
nothing else.

## Files

| File | Purpose |
|---|---|
| `supervisor.sh` | The mint → spawn → reap loop. `run` (default), `stop`, `status`. |
| `gh-app-mint.sh` | Mints short-lived GitHub App credentials (JWT → scoped installation token → JIT config). No PAT, ever. |
| `lib/pidtree.sh` | Process-tree tracking and kill-tree primitives `supervisor.sh` builds on. Read this file's header first — it documents the orphan bug this whole directory exists to fix. |
| `install.sh` | Idempotent box setup: checksum-pinned actions-runner download, directories, config seed, launchd plist render (never loaded automatically). |
| `box-runner.env.example` | Config template; `install.sh` seeds `box-runner.env` from it once. |
| `com.qwen38.box-runner.plist.template` | launchd plist template `install.sh` renders. |
| `test/test-box-runner.sh` | Dry-run suite exercising all of the above against mock processes; no network, no root, no real actions-runner. |

## The orphan bug this fixes (read this before changing spawn/stop)

The retired qwen/bench supervisor
(`supervisor.sh` in the operator's machine-scripts repository, read-only
reference, never modified by this repo) spawned the actions-runner with:

```sh
su - "${RUNNER_USER}" -c '... timeout ./run.sh'
```

`su -` starts a **new login session**. When launchd stops the supervisor
daemon (crash, `launchctl bootout`, an operator's kickstart), the signal
targets the **daemon's own process group** — and the `su`-spawned session
was never a member of it, so it is not reached even though both were alive
at that instant. The job's whole process tree becomes a live orphan that
outlives the daemon supposed to manage it, invisibly, until an operator
happens to notice it in `ps`. That is box-3-proven, not hypothetical.

**The fix**, in `lib/pidtree.sh` + `supervisor.sh`: the moment a job's root
pid is known, it is written to a state file (`state/job.pid`) — **before**
this process waits on it. `stop` reads that pid and walks the **live
parent-pid chain** from it (a kernel fact the kernel maintains independent
of session id or process group, so `su -`'s session boundary hides nothing
from it), signaling every descendant it finds. This works as a **standalone
entrypoint**: it does not need the process that originally spawned the job
to still be alive, because the persisted pid file, not an in-memory
variable, is the source of truth. `supervisor.sh run`'s own startup calls
the identical reap before minting anything new, so restarting the daemon
after an ungraceful death (the box-3 failure mode exactly) actually
recovers instead of leaking a second, concurrent job tree.

`tools/box-runner/test/test-box-runner.sh` proves this with a real detached
child (see "Tests" below) — both directly against `lib/pidtree.sh` and
end-to-end through `supervisor.sh`'s actual `run`/`stop`/`status`
entrypoints, including calling `stop` from a separate process invocation
than the one running the job loop (the actual recovery scenario).

**Known limitation.** The guarantee above holds whenever `stop` (or the
startup reap) runs *while the tracked root is still alive* — which is
exactly the box-3 scenario (the daemon dies, the job tree does not). If a
job instead forks a detached grandchild **and its own root process also
exits normally** before anyone calls `stop`, that grandchild's parent-pid
link is gone by the time anyone looks, and `stop` cannot find it by walking
from the (now-dead) recorded root. `supervisor.sh` still runs a defensive
kill-tree pass right after every job's root exits (`reap_between_jobs`),
which closes the common sub-case where the straggler is still alive in that
narrow window, but there is no unconditional guarantee against a fully
self-orphaned process from a misbehaving job step. This is documented
rather than hidden behind a heuristic process-table sweep that could
false-positive on an unrelated process; nothing in this directory scans by
command-line substring for that reason.

## Parameterization surface (env vars)

Everything below is read from `box-runner.env` (or the process environment,
which wins if both set the same var — `box-runner.env` is sourced early).

**Required, no default — `supervisor.sh` refuses to start without every one
of these (fail-closed):**

| Var | Meaning |
|---|---|
| `RUNNER_DIR` | Path to the actions-runner install (`install.sh` creates it). |
| `RUNNER_LABELS` | JIT registration labels. **Must match** the `runs-on:` list in `.github/workflows/benchmark.yml` — `macOS,qwen3.8-125b-a6b-cuda-v1` (`self-hosted` is force-included by `gh-app-mint.sh`). This is the entire coupling surface between the two sides — get it wrong and every dispatched job silently finds no runner. The seeded `box-runner.env` ships it empty and `supervisor.sh` refuses to start until an operator sets it. |
| `GH_APP_ID`, `GH_APP_KEY`, `GH_RUNNER_REPO` | Required by `gh-app-mint.sh` (see its own header); set via env or `BOX_RUNNER_APP_ENV`. |

**Optional (sane defaults, see `box-runner.env.example` for every one):**
`RUNNER_USER`, `GH_APP_MINT`, `RUNNER_NAME_PREFIX`, `LOG_DIR`, `STATE_DIR`,
`PAUSE_BETWEEN_JOBS`, `IDLE_RECYCLE_SECONDS`, `STOP_TERM_TIMEOUT_SECONDS`,
`ALERT_HOOK`, `GH_INSTALLATION_ID`, `GH_RUNNER_GROUP`, `GH_WORK_FOLDER`.

**Test/dry-run only — never set on a real box:**
`RUNNER_SU_ENABLED=0` skips the `su -` privilege drop (spawns `run.sh`
directly as the invoking user); `BOX_RUNNER_MAX_ITERATIONS=N` bounds the
loop instead of running forever. Both exist purely so
`test/test-box-runner.sh` can exercise the real spawn/stop/status/kill-tree
code path without root or a provisioned runner account. `install.sh` never
sets either.

## Design constraints and how they're met

- **Fail-closed everywhere.** Every required var above, plus `RUNNER_DIR/run.sh`
  and `GH_APP_MINT` existing and being executable, is checked before the
  loop starts; a miss exits 78 (`EX_CONFIG`) with a named message on stderr
  (and the log). `gh-app-mint.sh` independently fails closed on its own
  three required vars, so it refuses even when invoked outside
  `supervisor.sh`.
- **No secrets in code.** The GitHub App private key never appears in this
  directory; only its *path* (`GH_APP_KEY`) is configured, read once by
  `gh-app-mint.sh`. The JIT config crosses to the runner off argv, via
  `ACTIONS_RUNNER_INPUT_JITCONFIG` (the actions-runner's own env-input
  mechanism), so no other uid can scrape it from `ps` — carried over
  unchanged from the box-3 lineage.
- **Single job at a time.** JIT/ephemeral registration guarantees this by
  construction (the pattern is unchanged from the box-3 lineage); the loop
  is strictly sequential and only ever tracks one job's pid at a time.
- **Between-jobs quiescence, verifiably.** `supervisor.sh status` lists every
  live process in the current job's tree (or prints `idle`); a job's tree is
  reaped (`reap_between_jobs`) immediately after it exits and before the
  next mint.
- **Nothing here scores anything.** These scripts get the workflow a
  machine and nothing more — no benchmark logic, no measurement, no
  scoring. That lives in benchd, per this repository's own contract
  elsewhere in the tree.

## Tests

```sh
tools/box-runner/test/test-box-runner.sh [-v]
```

35 assertions (floor 27, checked by the suite itself so a section that stops
running silently under `set -uo pipefail` without `-e` cannot look green).
No network access, no root, no GitHub App credential, no real
actions-runner binary — every job in the suite is `test/mock-run.sh`, which
spawns a **detached grandchild** (`test/mock-orphan-child.sh`, under
`setsid` when available) exactly the way a job's own subprocess can outlive
`su -`'s session boundary in production. The suite's required regression
coverage:

- **Section 1b** kills a live mock tree via `pidtree_kill_tree` directly and
  asserts the detached grandchild is dead afterward — the fix.
- **Section 1c** is the negative control: killing *only* the tracked root
  pid (the naive/original behavior) leaves the grandchild running — proving
  1b is not a vacuous pass.
- **Section 2** repeats the same proof end-to-end through `supervisor.sh`'s
  real `run`/`stop`/`status` entrypoints, including calling `stop` from a
  **separate process invocation** than the one running the job loop — the
  actual box-3 recovery scenario, not just the library call.
- **Sections 3-4** assert every fail-closed refusal (missing `RUNNER_DIR`,
  `RUNNER_LABELS`, `GH_APP_MINT`, and `gh-app-mint.sh`'s own
  `GH_APP_ID`/`GH_APP_KEY`/`GH_RUNNER_REPO`) exits 78 with the expected
  named message.
- **Section 5** unit-tests `install.sh`'s pure helpers (checksum lookup,
  sha256 computation) in isolation, without touching the network.

Run `shellcheck -x tools/box-runner/*.sh tools/box-runner/lib/*.sh
tools/box-runner/test/*.sh` — clean with zero findings at any severity.
Every waiver in this directory is a targeted, commented
`# shellcheck disable=<code>` at the specific line, never a blanket
suppression; each one explains why the finding is a false positive for that
exact line (indirect invocation via `"$@"`/`trap`, or deliberate
non-expansion inside a nested `bash -c`).

## Operation

```sh
# One-time box setup (creates RUNNER_DIR, LOG_DIR, STATE_DIR, seeds
# box-runner.env, renders the plist -- does NOT install or load it):
tools/box-runner/install.sh

# Edit box-runner.env: set RUNNER_LABELS, and export/pin GH_APP_ID /
# GH_APP_KEY / GH_RUNNER_REPO (env or BOX_RUNNER_APP_ENV).

# Go live (operator, as root):
sudo cp tools/box-runner/com.qwen38.box-runner.plist /Library/LaunchDaemons/com.qwen38.box-runner.plist
sudo chown root:wheel /Library/LaunchDaemons/com.qwen38.box-runner.plist
sudo chmod 644 /Library/LaunchDaemons/com.qwen38.box-runner.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.qwen38.box-runner.plist

# Check what the box is doing right now:
tools/box-runner/supervisor.sh status

# Stop whatever job is in flight (safe any time, including after a crash):
tools/box-runner/supervisor.sh stop
```

## Restore / uninstall

```sh
# Take the daemon down. This sends its signal to the DAEMON's own process
# group only -- it will NOT reach an in-flight job's `su -`-spawned tree
# (the whole reason this directory exists). Always pair it with `stop`:
sudo launchctl bootout system/com.qwen38.box-runner
tools/box-runner/supervisor.sh stop

# Confirm quiescence:
tools/box-runner/supervisor.sh status   # expect: idle

# Full removal:
sudo rm /Library/LaunchDaemons/com.qwen38.box-runner.plist
rm -rf tools/box-runner/state tools/box-runner/log
# RUNNER_DIR (the actions-runner install itself) is left in place
# deliberately -- remove it by hand if the box is being fully decommissioned.
```

## What is unverified

Everything in this directory was exercised only against
`test/mock-run.sh`/`mock-orphan-child.sh` on a laptop (macOS, Apple
Silicon, no root, no real actions-runner binary). Not exercised, because
this task was laptop-only with no box contact:

- `install.sh`'s actual download-and-extract path against the real
  `actions-runner-osx-arm64-2.336.0.tar.gz` asset (the sha256 pin was
  copied from `gh api repos/actions/runner/releases/latest`'s published
  release body, not independently re-verified against a downloaded byte
  stream).
- `supervisor.sh` running as root with `RUNNER_SU_ENABLED=1` (the real
  `su - RUNNER_USER` path) against a provisioned `runner` account.
- `gh-app-mint.sh` against a real GitHub App installation (JWT signing,
  token minting/scoping/revocation, and `generate-jitconfig` were reviewed
  against the box-3-proven original but not run live -- no credential was
  available or appropriate to use for this task).
- The rendered launchd plist actually bootstrapping and surviving a real
  `launchctl bootout` / crash on macOS.
- A real registration against the labels
  `.github/workflows/benchmark.yml` now names. The value is settled (it is in
  that file's `runs-on:` and in `box-runner.env.example`'s comments), but no
  runner has been registered with it and no dispatch has been picked up.
