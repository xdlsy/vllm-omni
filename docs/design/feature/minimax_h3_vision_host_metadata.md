# MiniMax H3 vision host metadata experiments

These internal, opt-in candidates preserve the existing vision computation and
offloader-managed module calls. They do not enable sequence parallelism or
change reference-image resolution. All switches default to `0`; only `1`
enables a candidate. Device correctness and performance require validation on
the intended shapes, dtype, software stack, and hardware.

## Tensor position metadata

`VLLM_OMNI_H3_VISION_POS_TENSOR=1` retains interpolation indices and weights as
CPU tensors instead of converting them to Python scalar lists and reconstructing
tensors. It concatenates the pieces in the original order, converts indices to
long and weights to the embedding dtype on CPU, then transfers them to the
embedding device. The interpolation formulas, four-term addition order, spatial
merger ordering, and temporal repetition remain unchanged.

The switch is checked per forward. No request activations or metadata are
retained across requests. The original list-based path remains the default.

## Reuse matching position grids

`VLLM_OMNI_H3_VISION_POS_REUSE=1` computes interpolation indices and weights
once per distinct `(height, width)` in a vision forward. References with the
same spatial grid reuse those CPU tensors, while the original reference order
and each reference's temporal repetition are preserved. The cache is local to
the call, so configuration, dtype, weights, and grids cannot become stale across
requests. It stores geometry metadata rather than embedding outputs or
activations. This switch works independently of tensor position metadata.

## Share sequence lengths across blocks

`VLLM_OMNI_H3_VISION_LENGTHS=1` converts the existing cumulative sequence
boundaries into a host tuple once per vision forward. Every original block and
its Q/K/V splits share that tuple, reducing 81 length reads to one in a 27-block
tower. It preserves the original integer-boundary computation and the same
per-segment SDPA calls. The original module calls and offloader hooks remain in
place; only opt-in calls carry the new optional `sequence_lengths` keyword.

The tuple is not retained across requests. The default path continues reading
the original device boundaries in each attention call. Removing the reads does
not imply that the whole Attention startup gap disappears; end-to-end vision
timing must include the single preparation read before the block loop.

## Cross-request CPU geometry cache (experimental)

Set `VLLM_OMNI_H3_VISION_POS_CACHE=1` to cache final CPU interpolation indices
and dtype-converted weights in eval plus inference mode. It also selects the
tensor metadata path. The model-local LRU keeps at most 8 entries and 8 MiB;
oversized entries are computed without retention. Keys include the ordered
full grid, position-table size and dtype. Device/dtype moves clear the cache.
Learned embeddings are read afresh on every call; training/autograd use the
original path. Cache contents are disposable and omitted from serialization.

## CPU rotary coordinates (experimental)

`VLLM_OMNI_H3_VISION_ROTARY_CPU=1` builds integer row/column coordinates on CPU
in frame, merged-block, intra-block order and transfers one packed coordinate
tensor. Frequency-table arithmetic, lookup and cos/sin stay on the original
device. This is independent of the interpolation cache and does not cache
learned data. CPU exact-output checks cover rectangular, repeated and
multi-frame grids.

## Early first-block prefetch (experimental)

`VLLM_OMNI_H3_VISION_EARLY_PREFETCH=1` asks the first vision block's DLO hook
to start its current weights before PatchEmbed and position preparation. Only
rank-local group heads with shared-slot ownership tracking participate; other
layouts retain the original path. The original pre-forward hook still waits
for readiness and rechecks ownership, including synchronous restoration if
another group overwrites the slot. Collective ordering is unchanged. CPU
protocol tests cover async start, readiness wait and overwritten-slot fallback.

## Validation scope

Real-weight Ascend NPU checks covered individual candidates and their combined
path with rank-local DLO, including repeated, rectangular and multi-frame
grids. Both intermediate-checkpoint and final-output-only comparisons were
byte-identical for the tested inputs. Eight-rank service checks also matched
the vision checkpoints. These results do not establish correctness for every
device, shape, dtype or offloader layout.

The selected experimental combination enables the three switches above along
with tensor position metadata, matching-grid reuse and shared sequence lengths.
All switches remain off by default. Timing observations are exploratory; full
service output determinism has not passed, so these checks do not establish
end-to-end losslessness or an accepted end-to-end speedup.
