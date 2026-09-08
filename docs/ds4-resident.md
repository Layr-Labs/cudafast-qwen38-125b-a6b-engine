# The resident engine: one weight load for each benchmark window

## 1. The problem

benchd starts a new `cuda-engine` process for each phase. The phases are
warmup, timed prefill, timed decode and correctness. An official run repeats
this for each pair in each cohort, and for both legs.

When the engine is linked into the worker, each phase loads the checkpoint
again. An official run then does 16 loads or more. The artifact is 103.7 GiB.
The speculative leg adds a 2.8 GB draft head. The ranked pipeline must finish
in 20 minutes, so this is not possible.

The ruling is clear (David, 2026-08-30): the weights load ONCE for each window.
A persistent process holds them. The phases reconnect.

## 2. The decision

The weights get an owner: `ds4-resident`. It is one process. It opens the model
one time, and it opens the MTP draft head in the same process. It listens on a
Unix socket. Each phase's worker connects to it and loads nothing.

`tools/serve-up.sh` starts one resident for each window, inside the GPU-lock
window that the caller holds. It exports `DS4_RESIDENT_SOCKET`. Each
`cuda-engine` reads that variable and connects.

The resident never outlives the script that started it. The teardown runs on
success and on failure. There is no path that leaves it up: a resident holds
the GPU and the host memory of the artifact, so an orphan would block the box.

The measured cost of a reconnect is 0.266 ms (`tools/test-ds4-resident.sh`).

## 3. Why not the engine's own `ds4-server`

ds4 supplies `ds4-server`, and `make cuda-spark` already builds it. The build cost of that option is therefore zero. The option was
examined first, and it was refused. The reason is capability, not cost: its
wire cannot carry the contract that `harness/protocol-adapter/ds4_shim/ds4_shim.h`
states. Four parts of the contract have no equivalent.

Evidence, against pin `278b799` (unchanged from upstream `110afdd`: the port
adds no logit surface to `ds4_server.c`):

| What the adapter needs | What `ds4_server.c` has | Evidence |
| --- | --- | --- |
| The top 8 logits at each correctness step (`ds4s_top_logits`) | Nothing. The server has no logit surface at all. | `grep -c logit ds4_server.c` reports 0 over 19,551 lines. The request parser reads no `logprobs` key. |
| Token ids in and token ids out (`ds4s_sync`, `ds4s_eval`) | Text in, text out. The server does its own tokenization. | The request keys are `messages`, `prompt`, `input`, `stop`, `temperature`, `top_k`, `top_p`, `min_p`, `seed`, `max_tokens`. There is no `prompt_token_ids` and no `echo`. |
| A teacher-forced step: evaluate the caller's token, report the model's own choice, and do not feed it (`ds4s_eval`) | No such verb. A chat completion always feeds the token it selected. | `ds4_server.c:12470-12510`, the generation loop. |
| Per-cycle speculative counters (`ds4s_spec_counters`) | Nothing on the wire. Speculation is an internal choice. | `ds4_server.c:12484` selects the speculative path internally. The usage JSON at `ds4_server.c:6218` reports prompt, completion and cached tokens only. |

Two more facts support the refusal. The server disables MTP speculative
decoding when native session batching is active (`ds4_server.c:14332`), which
couples the scored mode to an unrelated switch. And the server holds a disk KV
cache with its own reuse policy, which benchd's drain contract cannot address.

A mapping onto that wire would have to reconstruct logits from text, retokenize
the prompt, and infer acceptance from token counts. Each of those steps is
lossy, and the scored numbers come from them. So the resident server is our
own: 500 lines of C over the same `ds4_shim.h` surface, in the same build.

## 4. The wire

NDJSON over `AF_UNIX`/`SOCK_STREAM`. One request object for each line. One
response object for each line. The order is strict. Each response carries `ok`.
A failed response also carries `error`, with the engine's own text.

One client at a time. The scored series is single-stream, and benchd runs one
phase at a time. A second connection waits in the listen backlog.

| Verb | Request fields | Response fields | `ds4_shim.h` call |
| --- | --- | --- | --- |
| `hello` | — | `vocab_size`, `eos_token`, `mtp_armed`, `draft_tokens`, `ctx_size`, `load_epoch`, `ident`, `model_path`, `mtp_head_path` | `ds4s_vocab_size`, `ds4s_eos_token` |
| `invalidate` | — | — | `ds4s_invalidate` |
| `sync` | `tokens` | `token` | `ds4s_sync` + `ds4s_argmax` |
| `eval` | `token` | `token` | `ds4s_eval` + `ds4s_argmax` |
| `argmax` | — | `token` | `ds4s_argmax` |
| `top_logits` | `k` | `ids`, `logits` | `ds4s_top_logits` |
| `eval_speculative` | `first_token`, `budget` | `tokens`, `token` | `ds4s_eval_speculative` + `ds4s_argmax` |
| `spec_counters` | — | `drafts`, `hits`, `quenches`, `disagreements` | `ds4s_spec_counters` |
| `bye` | — | — | — |

Each verb is one C function. Nothing is mapped, approximated or reconstructed.

`load_epoch` is the resident's pid. Each phase of one window reports the same
value, so an artifact can show that the load did not repeat.

Each verb that changes the state also returns the frontier argmax. The client
caches it, so `argmax` is free. This matters because the free-run loop is
inside benchd's timed window: a decoded token costs one round trip, not three.
The argmax is a pure read of logits that the same call left ready, so the eager
reply changes no semantics.

### The disagreement counter, and the order it must ship in

`spec_counters` carries a fourth value, `disagreements`. The engine raises it on
a REJECTING round where the batched verify's row-0 argmax and the one-row replay
of the same position chose different tokens. The tower is not batch-invariant,
so the two argmaxes can differ. The replay stands, because it is the row shape
the serial path would have run, and the divergence is counted. It never fails a
leg.

The worker reports the leg's delta on the `free_decode_run` response as
`verify_replay_disagreements`. benchd seals it as
`spec_verify_replay_disagreements`. It is audit-only. Nothing scores it and
nothing derives a rate from it.

The value is OPTIONAL, and an absent field means NOT REPORTED. That is a
different statement from `0`. The serial route reads no engine counters, so it
sends no such key rather than a zero it never measured.

**THE BOX'S benchd PAIR MUST BE AT `db3b73e` OR LATER BEFORE THIS ADAPTER IS
STAGED.** benchd added the field in pull request #258, at `db3b73e`. An OLDER
benchd parses the response with `deny_unknown_fields`, so it rejects the whole
line and the mtp leg fails.

This ordering cannot be enforced at run time. `hello` travels from the worker to
the benchmarker. The request envelope carries no benchmarker version, and there
is no handshake in the other direction, so the worker has no way to read which
benchd is in front of it. The constraint is therefore a DEPLOYMENT ORDER:
republish the dist channel first, resolve the new pair on the box
(`tools/fetch-benchd.sh`), then stage this adapter. The reverse order costs
every mtp leg until the pair is updated.

The two changes are independent in one direction only. A benchd at `db3b73e` or
later reads an OLDER worker, which sends no such key, without complaint. Only
the new worker in front of an old benchd fails.

The logits cross the socket as `%.9g`, which round-trips a `float` exactly.
The client then narrows the parsed decimal back to `float` before it widens it
to `double`. This is necessary. The linked backend widens the engine's `float`
with `logit as f64`, but a decimal parsed straight into `double` lands on the
nearest `double` to that decimal, which is a different number. The narrow step
recovers the engine's own `float`, so the socket path and the linked path give
identical `double` values.

## 5. The phase boundary

Each accepted connection is one benchd phase. The resident calls
`ds4s_invalidate` when it accepts, before the client's first byte. A phase can
therefore never inherit the live prefix of the phase before it. The worker's
own `drain_to_zero` invalidates again. That is idempotent, and it is deliberate.

A connection that sends nothing for `DS4_RESIDENT_PHASE_TIMEOUT_S` (default
1800) is dropped. The resident then accepts the next phase. A blocked phase
cannot hold the window's weights.

## 6. The memory plan

`tools/serve-up.sh` plans the memory before it starts the resident. It compares
the plan against `MemAvailable` in `SERVE_UP_MEMINFO` (default `/proc/meminfo`)
and prints it as a table. The plan is:

| Term | Source |
|---|---|
| body: every shard of the pinned body | the files on disk |
| **minus** the n-gram table | `per_layer_token_embd.weight`, sized from the artifact's GGUF tensor index by `tools/gguf-tensor-bytes.py` |
| plus the draft head, on the speculative leg | the file on disk |
| plus the session, KV and scratch | 4 GiB, for a context up to 8192 tokens |
| plus headroom | `DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES`, read from the pinned engine |

The n-gram table is **not** resident. The engine keeps it mapped on the
solid-state disk and streams it, and subtracts it from its own plan. For the
pinned target the table is 26.82 GiB of a 103.69 GiB body, so charging it wants
114.28 GiB where the true need is 93.46 GiB. On a 119 GiB Spark that refuses a
boot that fits.

The plan has no environment knob. The headroom comes from the engine, the
session budget is a constant in the script, and a context above the one the
budget covers is refused rather than under-charged.

The subtraction only holds for an engine that really streams the table, so the
plan reads the pinned engine's own declaration
(`ds4/ds4_qwen4exp.h`: `DS4_QWEN4EXP_MEM_PLE` and
`DS4_QWEN4EXP_MEMORY_HEADROOM_BYTES`) and REFUSES BY NAME when it is not there.
Against a pin with no qwen4exp port there is no plan, and nothing is loaded.
The plan also refuses when the table is absent from the index, and when its
type is not the one the track fixture pins
(`target.quantization.ple_table`).

If the plan does not fit, the script refuses and nothing is loaded. An OOM
during a load takes the box down. That happened on ai-server on 2026-08-29.

## 7. Fail-closed

`Ds4Session` has methods that cannot report a failure: `argmax`,
`top_logits`, `spec_counters` and `invalidate`. A failure in one of them
poisons the session. The next verb that can return an error returns it. In the
meantime `top_logits` returns nothing, and the backend already treats that as a
fault. A broken resident can therefore never read as a fast phase.

## 8. The identity strings

The `hello` of the Engine Protocol carries a `backend` string and a `device`
string. benchd seals them in the score as `engine_backend` and `engine_device`.
Nothing scores on them. They are the only place in a sealed artifact that says
which engine made the number.

Each of the three engine paths gives its own two strings.

| Path | `engine_backend` | `engine_device` |
| --- | --- | --- |
| The mock backend | `mock` | `none` |
| A worker attached to the resident | `ds4-resident load_epoch=<pid> <resident ident>` | `cuda sm_121` |
| The ds4 engine linked in the worker | `DS4_ENGINE_IDENT`, or `ds4` | `cuda sm_121` |

The `<resident ident>` part is the `DS4_ENGINE_IDENT` that `tools/serve-up.sh`
makes. It gives the engine pin, the nvcc release, the driver, the declared
depth and the draft tokens. An example of the full string is:

```
ds4-resident load_epoch=47842 ds4@278b799b nvcc=V13.0.88 driver=580.65 mtp1 draft_tokens=2
```

The mock announced `cuda` for the backend and `cuda` for the device until
2026-09-03. A mock artifact was thus the same as a real one in these two
fields. The mock says in words on its stderr that it is not inference, but the
official path does not forward the stderr of a worker. The strings above are
the remedy: a reader of the sealed artifact can now see the mock.

`load_epoch` is the pid of the resident. Each phase of one window connects to
the same resident, so each phase seals the same `load_epoch`. A second load
would make a second value. The one-load claim is thus in the sealed artifact.
Before this, the claim was only in the resident log and in
`serve-identity.json`, and benchd sealed neither.

## 9. What proves it

- `tools/test-ds4-resident.sh` runs the real serve script, the real resident
  source and the real adapter against a synthetic engine
  (`tools/ds4/resident-stub-engine.c`). It proves one load over four phases,
  the reconnect time, the session reset at each boundary, the counters, the
  memory refusal before any load, clean teardown, and the idle ceiling. It
  needs no GPU, no driver, no checkpoint and no network.
- `tools/test-serve-up-plan-memory.sh` proves the memory plan against synthetic
  GGUF tensor indexes: the subtraction to the kilobyte, the printed table, the
  headroom read from the engine, and the refusals when the table is absent,
  when its type is not the pinned one, when the engine declares no streamed
  table, and when the context is above what the session budget covers.
- `harness/protocol-adapter/src/tests.rs`, module `resident`, proves the client
  against a scripted server on a real socket: the verb mapping, the cached
  argmax, the poisoning, and partial reads.
- `tools/ranked-box-preflight.sh` refuses a repository where the fixture, the
  serve script and the build disagree about the weight owner.
