# The `free_decode_begin` / `free_decode_run` seam — adjudication (#109 W3 finding 6)

**Authority:** `PROTOCOL-v1.1.md` (SIGNED, David 2026-08-17) §2.1, §2.2, §2.4, §2.6, §3, and the
v1 loop §1.1 that §2.1 declares the begin's contract "identical" to.

**Verdict: the ENGINE is wrong. benchd's oracle is spec-true.**

Window 3 (`parity-window-20260820-3`) isolated the disagreement on the GPU against two independent
pinned tapes: the engine's `free_decode_run` stream was **0/16** against `emitted_tokens[0..]` and
**16/16** against `[reference_seed_token] + emitted_tokens`. The speculation is correct — the real
MTP head, depth 2, driving its own loop, token-exact against the reference's recorded greedy
continuation on both tapes. The two sides disagreed about **one index**, at the seam between `begin`
and `run`. This document records which side the spec puts it on, before any fix.

## 1. §2.2 gives the seed and the run stream two DIFFERENT oracle fields

The normative wire pseudo-code:

```
start clock
free_decode_begin(seed_tokens)               # -> seed_token
    verify seed_token == expected_decode_seed_token   # else TokenMismatch -> hard fail + session discard
free_decode_run(count = N)                   # -> { tokens[N], acceptance_lengths[], drafted_total, accepted_total, committed_total }
...
# benchd verification (external oracle check), all-or-nothing:
require tokens.len() == N
for i in 0..N:
    require tokens[i] == expected_decode_tokens[i]     # else TokenMismatch{step:i} -> hard fail + session discard
```

The seed token is verified **on its own line, against its own golden field**
(`expected_decode_seed_token`). `tokens[i]` is verified against a **different** field
(`expected_decode_tokens[i]`). For the engine's behaviour to be conformant,
`expected_decode_tokens[0]` would have to equal `expected_decode_seed_token` — the golden would have
to carry the seed twice. §1.1 says it does not.

## 2. §1.1 fixes what `expected_decode_tokens[0]` IS

v1's teacher-forced loop, which §2.1 says `free_decode_begin` has an "**identical contract**" to:

```
decode_begin(seed)                         -> verify seed_token == expected_decode_seed_token
for step in 0..N:
    input = (step == 0) ? expected_decode_seed_token
                        : expected_decode_tokens[step - 1]   # ORACLE token, forced
    token = decode_step(input)             -> verify token == expected_decode_tokens[step]
```

`expected_decode_tokens[0]` is the output of the step whose **input** is
`expected_decode_seed_token`. It is the token **after** the seed, by construction of the golden.
There is no reading under which the seed is also `expected_decode_tokens[0]`.

## 3. §2.1 / §2.4 — the begin COMMITS the seed; the run commits N MORE

§2.1, `free_decode_begin`:

> exactly one seed forward; **establishes the last-committed state**. Identical contract to v1
> `decode_begin`.

§2.1, `free_decode_run`:

> the engine free-runs its own MTP loop until it has **committed** N tokens, then returns all N
> materialized token IDs plus AUDIT counters.

The seed token is already the last-committed state when the run starts. The run's N are N *further*
commits. §2.4 says the same thing from the accounting side:

> Every committed token is either an accepted draft or a base-model fallback token; **the fallback
> token counts as drafted** (RULED, OQ5).

## 4. Does N count the seed? The spec's OWN arithmetic says no.

This is the question the audit triple settles, and it settles it against the engine. §2.6:

> For a v1.1 free-run decode phase, `completed_work` counts the **verify-round target forwards**:
> **one seed forward plus one forward per MTP verify round R**, so the **counter MUST equal
> `R + 1`** … benchd enforces this as a normative consistency TRIPLE:
> 1. `R == acceptance_lengths.len()`
> 2. `sum(acceptance_lengths) == N`
> 3. `completed_work == R + 1`

The seed forward is counted **separately** from the R verify rounds — it is the `+ 1`. And §3
defines each histogram entry as:

> per verify-round committed count: how many draft tokens survived internal verification **(plus
> fallback)** and were committed in each MTP round. Length = number of rounds R.

So `sum(acceptance_lengths) == N` sums the tokens committed **by the R verify rounds**. The seed
token is the product of the **seed forward**, not of any verify round. Therefore **N excludes the
seed**, and `tokens[]` — whose length must equal N and whose `committed_total` must equal N (§2.4) —
begins after it.

Note the triple *held* on window 3's responses (`acceptance_lengths=[3,1,3,3,3,1,1,1]`, sum 16 =
`committed_total` = `tokens.len()`). It had to: the engine emitted 16 tokens over 8 rounds either
way. The triple constrains **counts**, not **identities**, so it could not catch a pure one-position
relabelling. §2.2's per-index oracle check is the constraint that does, and that is the one that
fired.

## 5. Which implementations already match the spec

* **benchd's runner** — `crates/bench-runner/src/timing.rs::measure_free_run_decode`: verifies
  `seed_token` against `expected_decode_seed_token`, then `tokens[i]` against
  `expected_decode_tokens[i]`. Spec-true.
* **benchd's mock** — `crates/bench-runner/src/mock.rs`: `free_decode_begin` returns
  `oracle.seed_token`; `free_decode_run` returns `decode_tokens.take(N)`. Spec-true.
* **The engine's own SERIAL free-run route** — `runFreeDecode(route: .serial)`: each round is
  `plainDecodeStep(inputToken: lastCommitted)`, which returns the **next** token. It never re-emits
  the seed. **Spec-true — and it is benchd's paired DENOMINATOR.** The engine was internally
  inconsistent between the two routes of one wire verb, and the conformant one is the control leg.
  Had benchd moved its oracle instead, the serial control would have become the broken side.
* **The engine's MTP free-run route** — the only non-conformant implementation. Its round result is
  in **commit order** (`[primary] + acceptedDrafts`, the order KV rows are written), which lags
  **production order** by exactly one: the round's opening primary was determined by the *previous*
  target forward, and the fallback *this* forward produced is not in its own `tokens`. Emitting the
  commit list verbatim re-sends the token `free_decode_begin` already returned.

## 6. Why neither conformance kit caught it

* The engine's GPU-free kit (`RuntimeWorkerGenericDispatchTests`) fed the free-run builder rounds
  **directly**, so the begin/run boundary was never exercised.
* The captured cross-repo wire fixture (`engine-wire-v1.jsonl`) carried `hello`,
  `phase_diagnostics` and `free_decode_run` — but **no `free_decode_begin` line**, so the seam was
  not a pinned cross-repo fact.

Both are closed by the same change: the seam is now a named, fail-closed, unit-tested step in the
trusted harness, and the captured fixture carries a `free_decode_begin` line whose `seed_token`
(699) is asserted **not** to appear in the run's `tokens[]` (which start at 700) — pinned by the
same sha256 in both repositories.

## 7. The fix, and why the audit histogram does not move

§3's per-round count is "accepted drafts **plus fallback**" = `acceptedDrafts + 1`, which is exactly
the count the engine's round already reports. So the correction is a pure one-position **relabel**:

| round | engine emitted (commit order) | spec-true emission (production order) | count |
|---|---|---|---|
| i | `[P_i] + drafts_i` | `drafts_i + [P_{i+1}]` | `1 + acceptedDrafts_i` — **unchanged** |

`R`, `acceptance_lengths`, `sum == N`, `completed_work == R + 1`, `committed_total` and
`drafted_total >= accepted_total` are all unchanged. Only the identities shift by one — which is
precisely the "16/16 under a one-token shift" window 3 measured. Replayed as a permanent GPU-free
test on window 3's own recorded histogram (`w3f6WindowThreeIsolationIsTokenExactUnderTheSpecTrueSeam`).
