# MiniMax H3 Vision Sequence Parallelism

This document describes the MiniMax H3 visual encoder's opt-in Ulysses SP8
path, its shard and output contracts, and its bitwise validation tools.
The shared [sequence-parallel design](sequence_parallel.md) describes the
framework-level APIs. This model-owned path uses explicit encoder groups and
whole spatial merger groups rather than the generic DiT sharding hooks.

## Overview

The default H3 encoder runs a replicated visual tower on each participating
text-encoder TP rank. Text tensor parallelism does not automatically shard the
visual tower. Vision SP8 partitions visual tokens across the existing eight
encoder ranks while preserving the full features consumed by the text path.

The serving feature is experimental and disabled by default. Its validated
scope is the fixed six-image A5 workload. A2, SP16, other input shapes and
full-service performance require separate validation. CPU layout support for
additional parallel sizes does not establish serving support for those sizes.

## Motivation

Replicated vision repeats full-input PatchEmbed, projection and MLP work on
every encoder rank. Token sharding reduces the rows processed by these
operations. Attention still needs every valid token within its original image
segment, so Ulysses exchanges token ownership for head ownership during each
attention operation.

## Architecture

The implementation has three responsibilities:

| Component | Responsibility |
| --- | --- |
| `encoder.py` | Own visual weights, modules and encoder-group selection |
| `vision_parallel.py` | Build shard indices, exchange layouts and gather outputs |
| `vision_sp.py` | Execute the opt-in sharded visual forward through original blocks |

```text
Original pixels and image grids
    -> select local whole spatial merger groups
    -> local PatchEmbed and positional features
    -> repeat for each visual block:
         local norm, QKV and RoPE
         token-to-head All-to-All
         per-image attention on full valid sequences and local heads
         head-to-token All-to-All
         local output projection, residual and MLP
    -> local final / DeepStack mergers
    -> AllGather and restore original global merger order
    -> full final and DeepStack features on every text-encoder TP rank
```

All ranks retain the complete visual weights. The original block calls are
preserved so distributed layerwise-offload hooks can stage those weights.
This feature does not introduce visual weight tensor parallelism or change
text, VAE, DiT, request scheduling or downstream rank membership.

## Configuration and supported layouts

`VLLM_OMNI_H3_VISION_SP_SIZE` controls serving behavior:

| Value | Behavior |
| --- | --- |
| `0` or unset | Keep the existing replicated vision path |
| `8` | Use the existing text-encoder TP8 device group for visual SP8 |
| Other values | Reject the unsupported layout during initialization |

```bash
export VLLM_OMNI_H3_VISION_SP_SIZE=8
```

Retain the workload's existing service and request settings, including
`--text-encoder-tp-size 8` and its eight participating devices. To restore the
original path, set the variable to `0` before starting the workers.

Collectives use the encoder's explicit device group, not an implicitly chosen
global world. Increasing `num_gpus` alone does not implement a visual-SP16 /
text-TP8 split. That arrangement requires a separate participant and handoff
protocol; it is not supported by this serving switch.

Serving attention Batch and the diagnostic full-row compatibility guards are
disabled. The isolated probe has additional experiments; enabling them does
not enable or validate a corresponding serving feature.

## Shard planning contract

`build_vision_plan(lengths, sp_size, num_heads, merge_size)` returns an immutable
`VisionShardPlan` with:

- `segment_lengths`: valid token counts for each image/frame attention segment;
- `rank_token_indices`: global token rows assigned to each rank;
- `rank_merger_indices`: global output rows assigned to each rank after merging;
- `heads_per_rank`: the visual head count divided by the parallel size.

The planner accepts nonnegative segment lengths divisible by
`merge_size ** 2`, and supported parallel sizes that divide the head count.
It allocates consecutive whole merger groups independently within each segment.
Earlier ranks receive one additional group when a segment is not evenly
divisible. Empty shards are explicit. Since remainder allocation restarts at
rank zero for each segment, aggregate loads need not be exactly balanced.

The fixed six-image configuration has 27 visual blocks, width 1152, 16 heads,
head dimension 72, merger size 2 by 2, output width 5120, and zero-based
DeepStack layers 8, 16 and 24:

| Reference | Grid T x H x W | Visual tokens | Merger tokens |
| --- | --- | ---: | ---: |
| 1 | 1 x 120 x 68 | 8160 | 2040 |
| 2 | 1 x 120 x 68 | 8160 | 2040 |
| 3 | 1 x 40 x 22 | 880 | 220 |
| 4 | 1 x 30 x 22 | 660 | 165 |
| 5 | 1 x 30 x 22 | 660 | 165 |
| 6 | 1 x 66 x 116 | 7656 | 1914 |
| Total | | 26176 | 6544 |

These shapes are a fixed-case example, not additional preprocessing overrides.
The probe uses the same image preparation helper and processor as the pipeline.

## Attention, communication and output invariants

Between attention operations, each rank holds its assigned token rows with
all channels. The forward All-to-All gives it all valid tokens and only its
assigned heads. The inverse exchange restores its local token ownership.

Equal-sized transport buffers permit uneven shards without variable-split
collectives. Padding exists only in transport buffers. Unpacking removes it
and restores global token order before attention. Images remain independent;
attention never mixes tokens from separate image segments.

The committed implementation exchanges Q, K and V separately and exchanges
attention output once: four model All-to-All calls per block, or 108 calls
per rank for 27 blocks. Smoke-test calls and diagnostic reductions are not
part of that count.

For the fixed A5 case, short attention segments of at most 1024 tokens retain
at least eight independent SDPA heads by repeating heads and discarding the
duplicate outputs. This does not add tokens or cross-image attention. It is a
shape-specific compatibility policy, not a general backend equivalence claim.

Final and all three DeepStack outputs are gathered and restored using global
merger indices on every encoder rank. The text path receives the original
full-output contract rather than rank-local features.

### Per-forward index reuse

A `VisionAllToAll` instance retains token-index tensors only for its shard
plan and device during one visual forward. It reuses those indices across
Q/K/V reordering and the inverse operations. Device changes rebuild indices;
a new request creates a new transport. Activations and attention results are
never cached.

For the fixed eight-rank, 27-block case, index construction decreases from
864 tensors to eight per rank per forward. Retained token indices occupy
204.5 KiB per rank. Final output-gather indices are outside this optimization.
Packing, communication counts, arithmetic and output order stay unchanged.

## Bitwise validation tools

The isolated tools load only `model.visual.*` weights from the text-encoder
checkpoint, with strict key checking. They do not run Turbo, text decoding,
VAE, DiT or the HTTP service.

`vision_encoding_probe.py` provides inspection and original-SP1 repeatability.
It checks the frozen baseline manifest, six ordered images and prompt before
preprocessing; records actual source paths and digests, configuration and
weight hashes; and rejects unsupported execution modes or existing output
directories.

Forward hooks capture inputs, positions, segment boundaries, PatchEmbed,
each block's QKV, attention, MLP and output, mergers and final features.
`TensorCheckpoints` saves CPU tensors and a manifest. A subsequent pass checks
shape, dtype, raw bytes and finite values, including the sign of zero.
Missing, duplicate, unexpected, nonfinite or unequal checkpoints fail.
Reports include SHA256, maximum absolute difference and the first mismatch.

`vision_ulysses_probe.py` runs the actual distributed world. Rank zero creates
a fresh complete reference; each rank compares its assigned rows against that
reference. The fixed workload covers 123 checkpoints per rank. Each failure
is agreed collectively before continuing; launch and process-group timeouts
bound unexpected failures.

These CPU copies, hooks and dumps perturb execution and invalidate timing.
Same-process repeatability is only a first gate. A supported candidate needs
repeated original-versus-candidate checks and complete-output validation on
its actual device stack. Full-service tensor/media checks and performance
measurements are separate gates.

## Testing

Run the focused CPU contracts without model weights:

```bash
python -m pytest --noconftest -p no:cacheprovider -q \
  tests/diffusion/models/minimax_h3/test_vision_probe_contract.py \
  tests/diffusion/models/minimax_h3/test_vision_service.py \
  tests/diffusion/models/minimax_h3/test_vision_transport_cache.py \
  tests/diffusion/models/minimax_h3/test_vision_batch.py
```

An original visual inspection uses a frozen baseline directory containing
`case.json`, `manifest.sha256.json` and `prompt.txt`:

```bash
PYTHONPATH="$PWD" python benchmarks/minimax_h3/vision_encoding_probe.py \
  --case /path/to/frozen-baseline/case.json \
  --model-root /path/to/Ref2VA --refs-dir /path/to/six-images \
  --platform A5 --mode inspect --output /path/to/new-inspection
```

For SP8 bitwise validation, select only authorized devices with sufficient
resources and use a new output directory and an external execution bound:

```bash
PYTHONPATH="$PWD" timeout --signal=TERM --kill-after=15 1800 \
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  benchmarks/minimax_h3/vision_ulysses_probe.py \
  --case /path/to/frozen-baseline/case.json \
  --model-root /path/to/Ref2VA --refs-dir /path/to/six-images \
  --sp-size 8 --short-sdpa-min-heads 8 --output /path/to/new-sp8-run
```

Keep baseline inputs and numeric settings fixed. Store raw tensors on a disk
with sufficient free space; full per-layer references can require tens of GB.
Check repeated fingerprints, every rank's coverage and the final plus all
DeepStack outputs before using an independent timing carrier. Performance
comparisons require same-window controls and resource-interference records.

## Troubleshooting and limitations

- **Unsupported layout:** verify the SP8 flag and exact encoder TP8 group.
  Planner support for SP2/4/16 does not make those serving layouts available.
- **First mismatching checkpoint:** use the recorded shape and first mismatch
  to localize the earliest divergence; do not relax bitwise equality silently.
- **Short-segment differences:** revalidate the SDPA head-shape policy on the
  actual target backend. A5 equivalence does not imply A2 equivalence.
- **Missing or corrupt reference:** validate source/input hashes and the tensor
  manifest; a partial run is not a passing correctness result.
- **Transport failure:** inspect rank/group agreement and the first error;
  never treat a communication timeout as proof of a numerical defect.
- **Unexpected timing:** remove checkpoint hooks and CPU dumps before measuring.
  Isolated visual improvements do not establish pre-DiT or HTTP improvements.

## Reference implementations

- [H3 encoder](../../../vllm_omni/diffusion/models/minimax_h3/encoder.py)
- [Shard planning and transport](../../../vllm_omni/diffusion/models/minimax_h3/vision_parallel.py)
- [Serving visual SP8](../../../vllm_omni/diffusion/models/minimax_h3/vision_sp.py)
- [Original visual probe](../../../benchmarks/minimax_h3/vision_encoding_probe.py)
- [Distributed bitwise probe](../../../benchmarks/minimax_h3/vision_ulysses_probe.py)
