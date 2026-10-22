# H3 vision configuration and correctness probe

The SP1 diagnostic and isolated Ulysses prototype are separate entry points.
The serving model is unchanged. `vision_encoding_probe.py` only executes
`--sp-size 1 --batch-mode off`; other modes fail explicitly. The experimental
multi-rank runner below implements Ulysses, with attention batching disabled.

## Run contract

Use a complete frozen baseline directory (including `manifest.sha256.json`
and `prompt.txt`) and the six original reference files. Every baseline
manifest entry, image and prompt is SHA256 checked before preprocessing.
The fixed case is Ref2VA, unguided, output 1344 × 768. The probe calls the
same reference resize helper and processor as the pipeline; it does not
introduce a short-edge override. It loads only `model.visual.*` weights,
strictly, from the text encoder checkpoint. Turbo, text encoding, VAE and
DiT are downstream and are not executed by this isolated probe.

```bash
python benchmarks/minimax_h3/vision_encoding_probe.py \
  --case /path/to/frozen-baseline/case.json \
  --model-root /path/to/Ref2VA --refs-dir /path/to/six-images \
  --platform A5 --mode inspect --output /path/to/new-inspection

# First verify the authorized physical devices are healthy and idle.
# Use a new directory, and an externally bounded process (e.g. timeout 1200).
python benchmarks/minimax_h3/vision_encoding_probe.py \
  --case /path/to/frozen-baseline/case.json \
  --model-root /path/to/Ref2VA --refs-dir /path/to/six-images \
  --platform A5 --mode correctness --device npu:0 \
  --output /path/to/new-correctness
```

The source checkout must be the active Python import root. Do not install
over another user's environment. Each run refuses to overwrite its directory.
Store checkpoint tensors on a disk with enough free space; the real six-image
case can consume tens of GB. The first pass saves CPU tensor checkpoints;
the second compares them and checks reference integrity. Missing, duplicate,
unexpected, nonfinite or unequal checkpoints fail. Both tensor equality and
raw bytes must agree, including the sign of zero. Tiny errors do not pass.
Reports give shape, dtype, SHA256, first unequal element and maximum absolute
difference. Hooks are removed even when the forward fails.

Checkpoints cover input, positions and segment boundaries, patch embedding,
each block's QKV/attention/MLP/output, final merger and every DeepStack output.
These synchronous diagnostic hooks invalidate timing: never use these runs
as a performance measurement. Same-process repeatability is only the first
gate; it does not prove cross-configuration or complete-video equivalence.
A2 requires its own original-versus-candidate checks on the actual A2 stack.

## Runtime integration audit

At parent `783101992`, the current service cannot enable vision16/text8 by
changing `num_gpus` alone:

| Source | Current behavior | Required future change |
| --- | --- | --- |
| `diffusion/executor/multiproc_executor.py` (`num_workers`, worker loop) | Creates `num_gpus` workers | Explicit visual-only roles and startup agreement |
| `diffusion/worker/diffusion_worker.py` (`init_distributed_environment`) | Initializes one global world; worker metadata uses its CPU group | Separate control world from downstream execution groups |
| `diffusion/data.py` (`infer_data_parallel_size`) and `distributed/parallel_state.py` | Infers DP as world divided by non-DP dimensions | Reject implicit DP2 as a substitute for vision SP16 |
| `models/minimax_h3/pipeline_minimax_h3.py` (encoder construction and `encode_prompt`) | `rank < text_encoder_tp_size` gates both visual and text loading/execution | Split vision participant and text participant roles |
| Same pipeline, `_dit_rank_world`, input/error/result broadcasts and VAE distribution | Uses existing world/device groups and shared lifecycle | Audit every caller before restricting downstream to 0–7 |

The proposed protocol is PREPARE → VISION → FEATURE_HANDOFF → downstream
0–7 → DONE/ERROR. Ranks 8–15 must wait for bounded control messages while
the primary eight execute text/VAE/DiT; they must not enter those collectives.
All ranks must agree on failures before entering another collective. This
protocol and its serving integration remain unimplemented. The CPU
`validate_stage_layout` checks proposed memberships only; it does not verify
live groups. No runtime environment flags are enabled by this change.

## Verified shape contract (fixed six-image v1)

The actual deployed configuration has 27 blocks, width 1152, 16 heads,
head dimension 72, merger size 2 × 2, output width 5120 and zero-based
DeepStack layers 8, 16, 24. The six image grids and segment lengths are:

| Reference | Grid T × H × W | Tokens | Merger tokens |
| --- | --- | ---: | ---: |
| 1 | 1 × 120 × 68 | 8160 | 2040 |
| 2 | 1 × 120 × 68 | 8160 | 2040 |
| 3 | 1 × 40 × 22 | 880 | 220 |
| 4 | 1 × 30 × 22 | 660 | 165 |
| 5 | 1 × 30 × 22 | 660 | 165 |
| 6 | 1 × 66 × 116 | 7656 | 1914 |
| Total | | 26176 | 6544 |

There are two equal-length groups: references 1/2 and 4/5. Their attention
can be investigated independently; they must never attend across images.
Sixteen heads permit SP16 (one head per rank), but per-image merger counts
are not uniformly divisible by 16. Whole-group uneven shards must preserve
global positions and original valid K/V order; padding is not proven lossless.
The CPU index plan supports empty shards and preserves all merger groups.

CPU tests establish index and comparator behavior only. On the initial A5
attempt, config/preprocessing and strict visual-weight loading succeeded,
but device initialization failed with ACL 507899 and driver namespace
conflicts. Subsequent SP1 testing in a compatible task container passed all
123 points on A5. This is not A2 or performance acceptance.

## Isolated Ulysses prototype

```bash
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=15 1500 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/minimax_h3/vision_ulysses_probe.py \
  --case /path/to/frozen-baseline/case.json \
  --model-root /path/to/Ref2VA --refs-dir /path/to/six-images \
  --sp-size 2 --output /path/to/new-sp2-run
```

Select only authorized idle physical devices through the runtime environment;
local rank numbers refer to that visible list. Validate topology before running.
SP must match the actual world size (2/4/8/16), never fake multiple ranks per
device to claim scaling. Start with two same-domain devices. The process group
has a 900-second timeout; use a launcher timeout as shown. Comparison failures
are agreed by all ranks; other exceptions rely on launcher/group timeouts.

The runner first checks four rounds of distinguishable token/head transport,
including empty and uneven segments. Rank0 then saves a fresh full reference.
All ranks execute their assigned token rows through patch embedding, every
block and every merger. Q/K/V each undergo token-to-head All-to-All; Attention
retains original image boundaries; its output undergoes inverse All-to-All.
Hidden states remain sharded between blocks. Every rank retains the full
visual weights, with no text decoder or serving role instantiated.

Equal transport buffers contain padding only for communication. Valid indices
restore global token order before SDPA; padding never participates in attention.
Each rank checks its rows against the fresh reference and must cover all 123
checkpoint names, including inputs/positions/boundaries and all DeepStacks.
Any missing, duplicate, nonfinite or unequal point fails collectively. Final
local merger indices cover the global output; downstream feature gathering
and the 16-vision/8-downstream service protocol remain future work.

Per-rank JSON records participants, token/merger indices, head count, actual
A2A calls/bytes, source/config/weight digests and comparison results. A normal
27-block candidate uses 108 model A2A calls per rank; smoke-test A2A and
diagnostic MAX reductions are separate. CPU copies and checks make this a
correctness-only runner. Do not use its duration as a performance result.

On the fixed six-image A5 case, unguarded SP2 passed all 123 points on both
ranks. Unguarded SP4 failed at the first block's attention output. Replaying
the same saved Q/K/V on one device reproduced differences on short segments
when reducing the head count, without any communication. Thus SP2 success
does not establish SP4/8/16 bitwise equivalence.

The experimental `--short-sdpa-min-heads 8` option duplicates independent
heads up to eight for segments of at most 1024 tokens, then discards the
duplicate outputs. It never extends the token sequence or mixes images.
The default is zero (disabled). On the fixed six-image A5 case, SP4 and SP8
with this option each passed two full runs: 123 checkpoints per rank were
bitwise identical to the original reference, and candidate hashes matched
between repeated runs. This does not establish SP16, A2 or serving support.
The cutoff is an experimental
selection for this fixed case, not a general backend guarantee. This adds
redundant short-segment work and has no established performance benefit.

## Equal-length attention Batch experiment

The isolated runner accepts `--batch` (default off) and `--sp-size 1` for
single-device isolation. Images with identical BHSD shapes are stacked on
the batch axis, never concatenated along the token axis. Outputs return to
their original image positions. Fixed references 1/2 and 4/5 form two groups;
the other references remain separate. The original full reference remains
unchanged. Per-rank `attention_stats` records actual SDPA and batched calls.

Fixed six-image A5 SP1 Batch passed 123 checkpoints in two runs. The final
instrumented run recorded 108 SDPA calls, including 54 two-image calls,
versus 162 sequential calls across 27 blocks. CPU tests establish image
isolation and restoration only. Combined SP8 and performance remain separate
gates; timing from the checkpoint runner is not a benchmark result.

## Opt-in global row compatibility experiments

The diagnostic runner provides four independent switches, all disabled by
default: `--replicate-patch-embed`, `--preserve-projection-rows`,
`--preserve-merger-rows` and `--preserve-qkv-rows`. The first repeats the full
PatchEmbed batch. The row guards scatter local values into their original
global row positions, execute the full-size operation and select local outputs.
Merger selection preserves complete spatial merge groups.

These experiments address observed A3 shape-dependent bitwise differences.
They add redundant computation and memory; they are not a performance solution
or serving integration. With the first three guards enabled, the fixed case
passed two full SP8 Batch runs, each with 123 checkpoints per rank. The initial
SP16 run failed at block0 QKV. The QKV guard passed a local diagnostic for all
16 shard layouts, but a complete SP16 rerun remains pending. No SP16 or A2
acceptance is claimed. Revalidate every enabled combination on its target device.

## Experimental serving SP8

`VLLM_OMNI_H3_VISION_SP_SIZE=8` opts the H3 encoder into vision SP8 on
its existing text TP8 device group. The default `0` retains replicated vision;
other values and non-TP8 layouts fail during initialization. The vision weights
and original block calls remain in place so DLO hooks still stage each block.
Final and all DeepStack tensors are gathered in original merger order on every
encoder rank before the unchanged text path. Attention Batch and the diagnostic
global-row guards are disabled. Short segments retain the probe's minimum eight
SDPA heads by repeating independent heads and discarding duplicate outputs.

This is an experimental integration, requiring fixed-input bitwise checks and
full-service validation on each target platform and shape. Isolated A5 probe
results do not establish service, SP16, or A2 correctness or speedup. Test-only
capture and A/B switching are external harnesses, not serving API contracts.

### SP8 development checkpoint

The serving integration is a separate change from the isolated Batch and
global-row compatibility experiments. Keep those experiments disabled when
continuing the serving SP8 baseline. The integration depends on the existing
`vision_parallel.py` transport and layout helpers; it is not a standalone
replacement for the whole vision model.

The fixed `H3-REF2VA-TURBO-8CARD-v1` six-image case was checked on A5 with
original vision and SP8, each repeated twice in the full service. Each request
recorded 24 tensor checkpoints on all eight ranks, including final vision and
three DeepStack outputs, text hidden states, DiT conditioning, noise and
latents. There were 192 reference points and 576 subsequent bitwise comparisons.
All four MP4 files were byte-identical to the frozen baseline, SHA256
`3c92a506b4b440999c190d286e4addcbac9aee6547722117d8f7d0e59b598dc6`.
This is evidence for that fixed A5 case, not arbitrary inputs or A2/SP16.

The complete evidence archive contains 1003 files, with SHA256
`f132e13168cfc7021cd90a52dfb44e8d67b30d3ff5033339469a32cf3cb4c254`.
Archive and per-file hashes were verified after transfer. The deployed core
sources were checked against the committed implementation with normalized
text and AST equality. Raw media and tensor dumps are kept outside Git.

Full-service performance acceptance remains open: all five initial A/B/A
groups contained foreign NPU processes. A later bounded admission wait did
not obtain a stable idle window. Those timings must not be used as accepted
speedup evidence. Isolated vision timings do not establish pre-DiT or HTTP
latency benefits. Next, repeat only the non-capture performance groups in an
idle window, with resource observations bracketing each A/B/A group.

CPU contracts can be checked without loading model weights:

```bash
python -m pytest --noconftest -p no:cacheprovider -q \
  tests/diffusion/models/minimax_h3/test_vision_service.py \
  tests/diffusion/models/minimax_h3/test_vision_probe_contract.py \
  tests/diffusion/models/minimax_h3/test_vision_batch.py
```

Keep the fixed inputs, prompt, Turbo adapter and numeric request settings
unchanged. Future Batch changes require their own bitwise and performance
gates and a separate commit. Neither SP16 nor A2 has passed the serving gate.

### Per-forward transport index reuse

`VisionAllToAll` retains token index tensors for its fixed shard plan and
device. The same indices serve Q/K/V reordering and the inverse operation
throughout one vision forward. Changing the device rebuilds the indices;
creating the next request's transport creates a new cache. Activations and
attention results are never cached. Packing, collective calls, output order
and attention computation are unchanged.

For 27 blocks on eight ranks, this reduces token index tensor construction
from 864 to eight per rank per forward. The final output gather indices are
outside this change. On the fixed 26176-token case, retained indices occupy
204.5 KiB per rank until that forward's transport is released. CPU regression
coverage includes uneven and empty shards, changing input values, output
strides and construction counts at SP2/4/8/16. Run it with:

```bash
python -m pytest --noconftest -p no:cacheprovider -q \
  tests/diffusion/models/minimax_h3/test_vision_transport_cache.py
```

These layout tests do not establish NPU correctness or performance for a new
platform, input shape or parallel size; those still require runtime gates.
On the fixed six-image A5 case, SP8 with index reuse passed two complete
123-checkpoint runs per rank, with matching repeated tensor fingerprints.
The isolated timing carrier also checked the full final and three DeepStack
outputs on every rank before timing cached and uncached SP8. This does not
replace the full-service bitwise and performance gates for the changed source.
