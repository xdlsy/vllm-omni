# MiniMax-H3 encoder residency experiments

These eight experiments target `encode_prompt`, especially `encoder_offload`
and `device_memory_release`, while preserving encoder tensors bit for bit.
All environment options default to OFF. There is **no measured NPU speedup** in
this change. CPU lifecycle tests cannot establish NPU correctness or performance.
Use a fresh baseline on the target server before accepting any option.

## Scope and storage ownership

The options affect the unquantized MiniMax-H3 Qwen3-VL encoder, both through
pipeline-managed component staging and model-level sequential offload. They do
not change encoder math, TP collectives, DiT, VAE or final output encoding.
Keep encoder component offload enabled in the existing launch configuration.
A permanently resident encoder that never enters its placement lifecycle will
not exercise these options. Restart the service for every arm: options are read
once before its first encoder transition and every encoder rank must receive
identical environment settings.

Legacy encoder layerwise offload, HSDP, quantized encoder weights and cross-group
storage aliases are rejected. These experiments are not a training interface.
Frozen CPU masters and resident weights require immutable parameters **and
buffers**. Do not mutate/load weights or stateful buffers after the first
transition in these modes. For mutable inference state use copy-back modes;
`requires_grad=False` alone does not establish immutability.

| Option (prefix `VLLM_OMNI_ENCODER_`) | OFF | ON and actual behavior |
| --- | --- | --- |
| `FROZEN_MASTER` | `0` | `1`: snapshot final loaded weights once; offload rebinds CPU storage without D2H. |
| `CACHE_RETENTION` | `0` | `1`: retain allocator cache only when cached bytes ≤25% of capacity and physical free ≥5%; missing telemetry or failure flushes. Alone it keeps legacy weight copies. |
| `PINNED_BUFFERS` | `0` | `1`: allocate pinned CPU buffers once and reuse. Alone it still copies D2H; add frozen master only in a separately measured combination. |
| `RESIDENT_LAYERS` | `0` | `-1`: retain the entire encoder device weights; positive N: retain the first N text layers. Retained groups skip subsequent H2D and D2H. |
| `PACKED_TRANSPORT` | `0` | `1`: pack physical storage into an aligned byte buffer per staging group; one H2D/D2H transport per group, preserving aliases, strides and original offsets. |
| `GROUP_SIZE` | `0` (whole encoder) | `1`: one block; N>1: up to N consecutive blocks. Vision and text stacks are grouped separately. Nonblocks stay loaded until component exit. |
| `PREFETCH` | `0` | `1`: copy the next group on its copy stream while current group computes; requires explicit `PINNED_BUFFERS=1` and `GROUP_SIZE>=1`. Consumption waits on an event. |
| `REUSE_BUFFERS` | `0` | `1`: retain explicit device allocations, but copy CPU contents into them on every load; cannot combine with resident weights. |

Allocator cache is free, reusable space owned by the allocator. Resident weights
are live allocations whose existing contents are reused. Reusable buffers are
also live allocations, but their contents are refreshed. The last two consume
memory even after parameters are rebound to CPU. Reusable buffers are owned
**per staging group**, not shared across groups: after every group has run their
aggregate capacity can equal the whole encoder. This is an allocation experiment,
not a low-memory offload mode or a CUDA/NPU graph compatibility claim. Both modes
must pass downstream DiT/VAE/e2e peak-memory checks. Grouping cannot combine with
full (`-1`) residency; positive text-prefix residency can combine with grouping.
Packed initialization temporarily needs both source and packed host storage.

## Reproducible launch arms

Set `SERVICE_CMD` to the shell array containing the **existing, working server
launch command and arguments**, including its unchanged NPU/TP/offload options.
Do not add profiling or change sampling, prompt, dimensions, seed, steps,
concurrency or model weights between arms. Environment variables must reach the
worker processes (including every node for a distributed launch).

```bash
# Example form only: SERVICE_CMD=(bash /absolute/path/to/existing-launch.sh)
launch_arm() (
    arm="$1"
    shift
    unset VLLM_OMNI_ENCODER_FROZEN_MASTER VLLM_OMNI_ENCODER_CACHE_RETENTION
    unset VLLM_OMNI_ENCODER_PINNED_BUFFERS VLLM_OMNI_ENCODER_RESIDENT_LAYERS
    unset VLLM_OMNI_ENCODER_PACKED_TRANSPORT VLLM_OMNI_ENCODER_GROUP_SIZE
    unset VLLM_OMNI_ENCODER_PREFETCH VLLM_OMNI_ENCODER_REUSE_BUFFERS
    case "$arm" in
        baseline) ;;
        frozen) export VLLM_OMNI_ENCODER_FROZEN_MASTER=1 ;;
        cache) export VLLM_OMNI_ENCODER_CACHE_RETENTION=1 ;;
        pinned) export VLLM_OMNI_ENCODER_PINNED_BUFFERS=1 ;;
        resident-all) export VLLM_OMNI_ENCODER_RESIDENT_LAYERS=-1 ;;
        resident-prefix) export VLLM_OMNI_ENCODER_RESIDENT_LAYERS=4 ;;
        packed) export VLLM_OMNI_ENCODER_PACKED_TRANSPORT=1 ;;
        layer) export VLLM_OMNI_ENCODER_GROUP_SIZE=1 ;;
        blocks) export VLLM_OMNI_ENCODER_GROUP_SIZE=4 ;;
        prefetch-control|prefetch)
            export VLLM_OMNI_ENCODER_PINNED_BUFFERS=1
            export VLLM_OMNI_ENCODER_GROUP_SIZE=1
            if [ "$arm" = prefetch ]; then export VLLM_OMNI_ENCODER_PREFETCH=1; fi
            ;;
        reuse) export VLLM_OMNI_ENCODER_REUSE_BUFFERS=1 ;;
        *) echo "Unknown arm: $arm" >&2; exit 2 ;;
    esac
    exec "$@"
)
launch_arm baseline "${SERVICE_CMD[@]}"
# After stopping each service, run ONE of these and replay the identical request:
# launch_arm frozen "${SERVICE_CMD[@]}"
# launch_arm cache "${SERVICE_CMD[@]}"
# launch_arm pinned "${SERVICE_CMD[@]}"
# launch_arm resident-all "${SERVICE_CMD[@]}"
# launch_arm resident-prefix "${SERVICE_CMD[@]}"
# launch_arm packed "${SERVICE_CMD[@]}"
# launch_arm layer "${SERVICE_CMD[@]}"
# launch_arm blocks "${SERVICE_CMD[@]}"
# launch_arm prefetch-control "${SERVICE_CMD[@]}"
# launch_arm prefetch "${SERVICE_CMD[@]}"
# launch_arm reuse "${SERVICE_CMD[@]}"
```

Compare each independent arm with all-OFF baseline, **not** the preceding arm.
For prefetch, the causal control is `prefetch-control`; also compare both with
all-OFF to expose prerequisite costs. First validate single encoder rank, then
repeat each candidate with the original encoder TP size and rank placement.
Do not mix switches across ranks. Only test combinations after individual arms
pass; record all enabled settings and their own same-window control.

## NPU acceptance procedure

1. Record commit SHA, working-tree diff, checkpoint identity, torch/torch_npu,
   vLLM/CANN/driver versions, card IDs/topology, TP sizes, free memory and competing
   processes. Freeze requests, seeds, output conversion and synchronization
   boundaries. Confirm HTTP 200, response byte count and matching request log.
2. Warm up outside the profiler (at least 5 complete requests, at least 10 when
   compilation is enabled). Record cold initialization separately; pinning and
   packing startup costs must not disappear from the report.
3. In one controlled window alternate A/B/A, with at least three measured
   requests per arm. Use identical warmup and request count after each restart.
   Report samples, drift and noise floor; expand the window/sample count if the
   apparent benefit is within noise. Do not accept cross-day absolute values.
4. Collect unprofiled wall time first, then a separate NPU profile for the same
   workload. Profiler runs diagnose engagement; they do not replace wall time.
   Instrument the original stage boundaries identically in both arms, including
   required device completion. Group movement occurs inside `encoder_forward`;
   compare the **whole `encode_prompt` sum**, not merely the shrunken exit cost.
5. At each boundary collect process RSS/PSS, pinned/locked host bytes (platform
   counters and `/proc/PID/status` `VmLck`/`VmPin` where available; report unavailable
   values explicitly), allocated/reserved device bytes, physical free bytes and
   per-request peak allocated/reserved. Report every encoder rank, not only rank 0.
   Continue measurements through DiT, VAE and e2e; reject OOM or e2e regressions.
6. Save encoder hidden tensors before `result_to_cpu`/distribution using the same
   capture path in both arms. Require dtype/shape and bitwise equality on every
   rank for baseline reruns, option reruns and baseline-vs-option. Preserve raw
   bytes for comparison (including NaN payloads). Require unchanged final image/
   video file MD5, and inspect the output for pre-existing corruption; MD5 alone
   does not establish a healthy baseline. Any mismatch blocks acceptance.
7. Repeat text-only and image/video prompts, short/long sequence lengths, changing
   shapes, repeated identical requests and alternating requests (at least 20
   cycles). Exercise skipped vision, partial block groups and multiple encoder
   TP ranks. Test controlled OOM, copy failure, forward failure in the middle of
   a group, and cancellation before consuming a prefetch. Verify cleanup, next
   request correctness, no pinned/device-memory growth, and hook removal on
   sequential-offload disable/re-enable. Do not inject fatal device faults into
   a shared production server.

Use these exact stage columns. The reference seconds below came from the
original report and are **not measurements of this branch**:

| Stage | Reference seconds | Fresh A | B | Fresh A repeat |
| --- | ---: | --- | --- | --- |
| `input_prepare_and_distribute` | 0.656797 | | | |
| `encoder_load` | 0.614717 | | | |
| `encoder_forward` | 0.339266 | | | |
| `result_to_cpu` | 0.228486 | | | |
| `encoder_offload` | 1.132678 | | | |
| `device_memory_release` | 0.524752 | | | |
| `result_distribute` | 0.027588 | | | |
| `encode_prompt` | 3.524284 | | | |

Report DiT, VAE, output encoding and e2e beside this table. A reduction in
`encoder_offload` or `device_memory_release` alone is not acceptance.

## Prove engagement in the profile

Enabled runs log `encoder residency options=... lifecycle=... transport=...`.
Take per-request deltas of the cumulative H2D/D2H bytes and copy counts,
`device_allocations` and `buffer_reuses`. Cache-only has no stager transport
counters; instrument actual allocator telemetry and copies. Counters count byte
storage transports (including packing alignment), not parameter count. Verify
them against NPU memcpy events; counters are not proof of overlap or speedup.

- Frozen master: weight D2H bytes/count disappear after initial snapshot;
  activation `result_to_cpu` copies remain.
- Cache: fewer `FreePhysical`/`UnmapMem` events inside the target memory-release
  window; crossing either bound or missing telemetry still flushes. Check later
  stages, which retain their own cache-release policies.
- Pinned: host buffer addresses/pinned bytes stabilize after initialization;
  ordinary D2H remains when frozen master is OFF.
- Resident: selected device addresses and allocated bytes remain; selected H2D
  and D2H cease after initial load. Verify remaining groups still transfer.
- Packed: fewer and larger memcpy events, matching packed byte totals; alias and
  noncontiguous tensor checks pass. Count both transport directions.
- Grouping: trace shows groups loaded before their first block and unloaded after
  their last, with bounded simultaneous live groups (plus nonblocks/residents).
- Prefetch: inspect copy/compute stream lanes and event dependencies. Copy for
  group N+1 must overlap compute for N and finish before N+1 consumes weights.
  Group-exit synchronization remains; overlap and net benefit must be measured.
- Reuse: allocator calls stop after each group's first allocation, device
  addresses stay fixed, `buffer_reuses` rises, but H2D bytes repeat every load.
  Distinguish this from a warm allocator cache and from resident no-copy weights.

## Local CPU checks and remaining gates

```bash
python3 tools/run_encoder_residency_cpu_tests.py
```

The isolated runner uses real PyTorch tensors, views, aliases, rebinding and
forwards. It substitutes vLLM package initialization/logging and platform
stream/event operations; encoder/sequential lifecycle methods are loaded from
source without importing model kernels. Existing shared-stager tests run too.
Pin allocation is mocked because this CPU host cannot pin accelerator memory.
These tests cannot verify NPU DMA/event semantics, distributed collectives,
full-model output equivalence, memory capacity or latency. Run the normal
repository tests and local pre-commit hooks in the complete target environment,
then execute the NPU acceptance procedure before selecting a default.

Local validation for this change: Python 3.13 / torch 2.10.0 CPU, **42 passed**
(25 encoder experiment cases and 17 existing shared-stager cases). Ruff,
SPDX, test marks, forbidden-import, direct-device-call checks and diff whitespace
checks passed. The normal `python3 -m pytest --maxfail=1` entry point could not
initialize `tests.helpers.fixtures.config` because vLLM is not installed.
The full pre-commit bundle, mypy and Markdown lint are unavailable on this host;
these remain target-environment gates along with all NPU/full-model tests.
