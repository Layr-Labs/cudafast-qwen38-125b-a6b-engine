# Qwen4-Exp parity against the ds4-metal reference

## Purpose

This document records how the Qwen4-Exp operations in this engine compare with
a second, independent engine for the same model.

The reference is `ivanfioravanti/ds4-metal`, branch `qwen3.8-flash-next`, MIT
licence. It is a parity reference only. It is not a source of code.

## How the comparison is made

The reference tests each Metal kernel against a scalar host reference in the
test file. A kernel result is not ground truth. Two things transfer:

- the fixture inputs, which are deterministic formulas;
- the scalar reference, which states the same operation independently.

`make test-qwen4exp-parity` runs their scalar reference and our scalar
reference on their inputs. It needs no GPU and no model file. Our kernels are
tied to our scalar reference by `make test-qwen4exp-hc`, `-moe`, `-gdn`,
`-qsa` and `-ple-kernels`. The two steps together compare three
implementations: their reference, our reference, and our kernel.

The PLE hash is different. The reference publishes exact integer row ids, and
integers have no band. `make test-qwen4exp-ple` compares three
implementations directly: theirs, our carried history, and our scan rule.

## Agreements

Each line is one operation on which the three implementations agree.

| Operation | Evidence |
|---|---|
| PLE n-gram row ids | 80 of 80 ids exact, `tests/test_qwen4exp_ple.c` |
| PLE hash constants, head vocabularies and table size | exact, same values |
| PLE end-of-sequence rule, and a pass split on that token | exact |
| Q8_0 block matmul and block directory layout | 2.3e-6 relative |
| SiLU and SwiGLU, and the gate is the first matrix | 1e-9 relative |
| Q8_0 embedding gather, and the seed of each stream | exact |
| Hyper-connection mix: layout, gate position, mean over streams | exact |
| Hyper-connection write: reduction axis, factor of two, fused write | exact |
| Grouped RMS norm: grouping, epsilon position, weight index | exact |
| Router: top ten, tie to the lower index, softmax over the selected | ids exact, weights 1e-7 |
| Gated delta net convolution, activation and split | exact |
| Gated delta net decay gate and beta gate, with no floor | exact |
| Gated delta net delta rule and read-out order | exact |
| Gated delta net gated output norm | 2.4e-7 relative |
| Sparse attention output, with the key-value load share | 7.5e-9 relative |
| Sparse attention `complete` is integer floor division | exact |

## Differences

Each line is one difference. `make test-qwen4exp-parity` pins every one of
them, shows it is the only difference in that operation, and carries a
negative control.

| # | Difference | Reference | This engine | Stronger evidence |
|---|---|---|---|---|
| 1 | Hyper-connection inject divide | no divide | divides by the stream count before the sigmoid | this engine, from the MLX runner |
| 2 | Low-rank activation scale | no scale | scales by one over the stream count | this engine, from the MLX runner |
| 3 | Normalised value cast | stays f32 | rounds to bf16 before the weight multiply | this engine, because MLX casts there |
| 4 | Negative token id | gathers zeros for the vision path | no such case, this engine is text only | neither, the models differ |
| 5 | Gated delta net q/k norm epsilon | on the sum of squares | on the sum of squares, since this was closed | closed, see below |
| 6 | Gated delta net key-head pairing | grouped only | selectable, and production uses tiled | this engine, which tests both |
| 7 | Gated delta net token mask | a first-class mask for partial commit | no mask | the reference |
| 8 | Attention softmax normalisation order | divides each weight before the sum | divides the sum once at the end | neither, rounding only |
| 9 | Attention gate | divides by one plus the exponential | multiplies by the sigmoid | neither, the same value |

## Difference 5 is closed

Difference 5 was the only one that changed numbers by more than rounding, and
it is now settled against the original model.

`modeling_qwen4_exp.py` normalises the query and the key with
`l2norm(x, eps=1e-6) = x * rsqrt((x * x).sum(-1) + eps)`, on both the chunked
and the recurrent path, and then scales the query by `head_dim ** -0.5`. The
epsilon is on the SUM. This engine divided the sum by 128 first and then
scaled, which is the same expression with an epsilon 128 times larger on the
sum.

On the reference's own fixture the old form moved the key by 2.2e-3 and the
query by 2.0e-4. The key has unit length over 128 dimensions, so its
root-mean-square element is 0.088 and the key move is about 2.5 percent of a
typical element, near 600 times the reference's 2e-6 kernel band. Each engine
would have failed the other's test.

The engine now uses the sum form on both backends, with the query taking the
same 2^-3.5 constant it always did. `make test-qwen4exp-parity` asserts the key
agrees exactly and the query inside the reference band, and carries the old
form as a negative control that still measures 2.2e-3, so a return to it goes
red. `make test-qwen4exp-gdn` pins the kernel itself.

The MLX runner disagrees with the original model on this operation. It is not
cited here and the point is being raised with its owner.

## The MTP leg is scored against the serial leg, exactly

The engine drafts at depths 1, 2 and 3. Per round the head builds a chain of
*N* tokens by re-entering itself on its own hyper-connection output, the target
verifies `[fed token, draft_0 .. draft_{N-1}]` in **one forward of N + 1
rows**, the drafts the target's own greedy argmax confirms are accepted as a
prefix, and the round commits between 1 and N + 1 tokens. The chain is linear,
so there is no tree and no tree verifier: the accept is the longest matching
prefix of a single sequence.

**The contract is byte-identity with this engine's own depth-0 serial stream,
at every depth.** The goldens are authored from the serial path and the scored
benchmark reads the depth-N stream against them, so one differing token is a
scoring failure, not a quality question. Identity with llama.cpp is *not* the
contract and never was: it runs a different quantisation path.

This replaces the earlier framing on this page, which read serial identity as a
diagnostic tripwire and the per-depth `mtp1` oracle golden as the scoring truth.
That framing was correct while the tower's batch-shape residual was non-zero,
when a correct engine could have failed a serial comparison. The residual is
now zero, and it is zero by construction rather than by measurement, so the
stronger requirement is the one that holds.

### Why identity is reachable

Two ops chose a reduction strategy by row count:

- `ds4_gpu_matmul_q8_0_tensor` (`ds4_metal.m:18193`, against its decode-order
  entry at `:18263`), used by every Q8_0 projection, the HC mixer and the PLE
  block;
- `ds4_gpu_matmul_f32_tensor`, which takes **cuBLAS SGEMM above one row** on
  CUDA and a matvec at one row on Metal, used by the gated delta net's alpha
  and beta projections and by the MoE router's logits.

Every other op in the tower runs one threadgroup or block per token, so its
rows never see the batch. Both tiered ops are now routed inside the
speculative cycle's width, so that row *t* of an n-row call **is** a one-row
call -- by construction, not by measurement.

### The CUDA dense projection is one kernel at every width

On CUDA the Q8_0 dense projections -- the gated delta net and sparse attention
in-projections, the attention output projection, the hyper-connection mixers,
the per-layer embedding projections and the LM head -- run on an int8
tensor-core GEMM, `matmul_q8_0_preq_rows_mma_kernel` in `ds4_cuda.cu`, rather
than on the eight-row tile. `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
takes k = 32, which is exactly one Q8_0 group, so the group's dot is the same
exact int32 the older kernel's `dp4a` computes. Over groups the accumulation is
ascending, one f32 accumulator per output element, with no split-K and no warp
butterfly. The routed and shared expert projections use that same order.

It is the SAME kernel at every width. Rows past the call's row count are staged
as zeros and dropped on the way out, so width 1 is the width-512 kernel with a
padded tile and not a second path; and the tile SHAPE does not enter the
arithmetic, so the tile is chosen for occupancy without touching a number.
There is no routing left that could tier by row count.

This does change the dense tier's values against the eight-row tile, which
accumulated lane-strided partial sums and then a warp butterfly. The two
backends now differ there: Metal has no exact int8 MMA and keeps its own tiles.
Both stay row-invariant, which is the property the speculative cycle needs.
Identity BETWEEN the backends is not one of this engine's contracts; the
per-operation bands in the table above are.

From identical state with no rollback between, the difference is **zero** on
the pre-final-mixer row and zero on the logits, on Metal and on CUDA. CUDA
measured 2.37e-2 before the f32 half of that routing, and Metal 1.05e-3.
`make test-qwen4exp-graph` asserts both are exactly zero, so an op that starts
tiering by row count again goes red there. Depth 3 verifies four rows, so the
invariance sweep covers every width from 2 to 4.

### Rejected rows leave no trace

A round that accepts fewer than all *N* drafts rewinds **every** carried object
to the round's start position and replays the accepted prefix as a committed
pass of its own width. Rewind-and-replay rather than a partial rollback,
because the snapshot-class objects -- the GDN recurrent state and both
convolutions -- are running state with a single snapshot point: there is no
restore to the accepted length to be had, and a shorter forward is the only
thing that reaches it. Row invariance is what makes the replay bit-identical
to the rows it replaces.

The head's own cache is unwound the same way. At depth 2 and up the chain
writes cache rows *above* the frontier, from the head's hidden state rather
than the target's; those rows are dropped before the next round seeds the
committed ones, on the accepting path as much as the rejecting one.

`test_qwen4exp_mtp` compares every carried object against a serial shadow after
each round, so a round that leaves a speculative row behind is named on the
round that left it rather than on the token that eventually moved. Its
mutation proofs include two **one-row** rollback bugs -- a KV truncate and a
head-cache truncate that each stop one row short -- because a rollback that
lands one row past the accepted length is a different failure from one that
does nothing at all.

### The runtime counter

On a rejecting round the cycle compares the wide verify's argmax at the
mismatching row against the narrower replay's; where they differ it increments
`verify_replay_disagreements` and carries on. The replay's answer stands,
because the replay's distribution is what the next fed token is sampled from,
so the emitted stream and the fed tokens cannot drift apart.

The cycle refuses on neither, because a disagreement has two causes it cannot
separate from where it stands: a numeric residual, which must not kill a leg,
or a rollback object that failed to restore, which is a real fault. The suite
does not have that problem -- it knows the residual is zero -- so the tests
require the counter to be exactly 0 while the cycle only reports it. The
counter rides the shim's spec_counters path next to drafts and hits, so benchd
can seal a ceiling on that rate.

### The depth envelope

`DS4_MTP_DRAFT_TOKENS` counts the fed token, so it is the depth plus one: 1 is
a serial leg, 2 to 4 are depths 1 to 3, and 5 and beyond are refused by name.
benchd sends `{"mode":"mtp","mtp":{"depth":N}}` and the adapter maps it through
`ds4_qwen4exp_mtp_depth_from_draft_tokens`; there is no second knob.

## What the reference has and this engine does not

These are not disagreements. They are operations this engine does not carry.

- A token mask for the gated delta net, which the partial-commit path needs.
- Capture and replay of the convolution state and the recurrent state.
- A bf16 recurrent state, with the rule `bf16 == round(f32)` at the boundary.
- Multi-axis rope, and the vision path that uses it.
- bf16 key, value and pooled-index caches.

## Licence note

The reference's sparse attention Metal kernel is attributed in its repository
to `jundot/omlx` PR #3244. That licence is not verified. No part of that
kernel's structure is used here. Only the reference's scalar arithmetic and
its numbers are compared.
