# Profiling

> **Warning:** Profiling is for development and debugging only. It adds significant overhead and should not be enabled in production.

## MiniMax-H3 prompt-encoding stage timing

For MiniMax-H3 NPU tuning, add `--enable-diffusion-pipeline-profiler` to your
existing working server command, or set `enable_diffusion_pipeline_profiler=True`
in its diffusion configuration. This reuses the existing pipeline-profiler switch;
it does not require `profiler_config` or kernel trace collection.

Each local `encode_prompt` call emits one `[MiniMaxH3EncodePromptProfiler]` JSON
log containing `rank`, `status` (`ok` or `failed`), `timing_mode=host_wall`,
and the stages that actually ran (in seconds):

| Field | Measured boundary |
| --- | --- |
| `input_prepare_and_distribute` | Rank-0 image/video processing, token presentation, error propagation, and input distribution to Encoder TP ranks. |
| `encoder_load` | Encoder weight loading, through either the component lifecycle or the sequential offload hook. |
| `encoder_forward` | Qwen3-VL `_encode`, including input device copies and any layerwise weight staging performed inside forward. |
| `result_to_cpu` | `hidden.cpu()`, including any dependency waiting inherent in that copy. |
| `encoder_offload` | Encoder weight offloading performed inside this call, excluding separately timed cache release. |
| `device_memory_release` | Explicit Encoder cache-release calls; it does not include every allocator operation inside layerwise stagers. |
| `result_distribute` | Final hidden-state and token-tag broadcasts to DiT ranks, including waiting for the source rank. |

The same sub-stage durations accumulate under
`MiniMaxH3Pipeline.encode_prompt.<field>` in the existing per-request
`stage_durations`. The existing `MiniMaxH3Pipeline.encode_prompt` total remains
available. Its original outer wrapper still synchronizes before and after the
call. It includes the final device-completion wait and publishing the JSON
summary. No additional total-duration metric is introduced.

The new stages measure host wall-clock time in seconds and **add no device
synchronization**. Existing code is unchanged: the outer pipeline profiler still
synchronizes at its original function boundaries, and model-required
synchronization, blocking copies, and dependency waits remain in place.
No cross-rank barrier or reduction is added. Thus enabling the existing profiler
still has its original synchronization cost; these sub-stage timers add none.

Timing is off by default. When off, the new scopes do not read the clock or emit
summaries. When on, reading the clock, accumulating durations, and publishing
one summary per call still have overhead; this is not a zero-overhead guarantee.
Nested stages use exclusive accounting, so an explicitly timed cache release is
not counted again in `encoder_offload`.

Asynchronous operations can return before device execution finishes. In that
case `encoder_forward` measures host submission rather than complete NPU compute
time, and an existing blocking `result_to_cpu` can include waiting for forward.
These values describe the running host path and must not be interpreted as
isolated device-kernel durations. Use a separate kernel trace when that
distinction is needed.

Interpret the measurements using the actual offload configuration:

- Non-Encoder ranks report preparation and result distribution only; their
  result-distribution interval can include waiting for the entire Encoder run.
- Missing fields mean the stage was not entered, rather than a measured zero.
  A resident Encoder has no offload/release fields. Sequential model offload can
  defer Encoder eviction until a later stage; timing does not move it into
  `encode_prompt`. A load call on already resident weights may be very short.
- Layerwise transfers and internal stager cleanup remain charged to the enclosing
  forward/load/offload interval. `device_memory_release` covers only the explicit
  release scopes in the table.
- Other-component eviction (for example swapping out the DiT before Encoder
  loading), dispatch, and instrumentation overhead can make the total larger
  than the sum of the seven fields. Sub-stage values and the parent total must
  not be added together. Do not sum times across ranks.
- Failed calls report partial timing with `status=failed`; exclude those samples
  from successful-request timing. Each JSON summary belongs to one call, while
  `stage_durations` accumulates calls until the existing request-record reset.

Use these measurements to locate host-visible costs without adding synchronization
between the existing boundaries. Confirm the instrumentation's latency impact with profiling enabled
and disabled on the target NPU under the same workload. Recollect a baseline
with the same timing mode before comparing historical numbers; synchronized and
unsynchronized stage values have different attribution. CPU tests cannot
establish NPU output equivalence or negligible performance impact.

### Comparing an optimization with A/B/A

Use two separate measurement passes. The primary pass compares complete request
latency with pipeline profiling and kernel tracing disabled. The diagnostic pass
uses the pipeline-profiler switch identically in all arms and compares the
existing synchronized `MiniMaxH3Pipeline.encode_prompt` total. The JSON summary's seven sub-stages explain host-path changes; they are not the
acceptance metric for device work. Do not compare the two timing
modes or add parent and child durations together.

Reuse the existing [diffusion serving benchmark](../../benchmarks/diffusion/README.md).
Its `--output-file` JSON contains `latency_mean`, `completed_requests`,
`failed_requests`, and `stage_durations_mean`. For `/v1/videos`, its request
latency includes job submission, completion polling, content retrieval, and
job-cleanup time. Keep the same endpoint and client behavior across arms; polling
can obscure small changes. This is a client-visible metric, not device execution
time. In the diagnostic pass, use server logs if a deployment does not return
stage metrics in its response; do not treat an empty stage dictionary as zero.

1. Record the baseline and optimized revisions, exact launch commands, card group,
   weights, offload policy, and environment. Keep prompt/media, seed, resolution,
   frames, steps, concurrency, and request sequence identical. The optimization
   under test should be the only intentional difference.
2. Run baseline-before, optimized, then baseline-after serially in the same
   measurement window. Stop only the service processes you started between arms;
   check that the next launch actually uses the intended configuration.
3. After each restart, prewarm with at least five **full-workload** requests
   (at least ten with compilation), separately from the measured requests.
   The benchmark's built-in video warmup uses one frame and reduced steps;
   it is not a substitute for warming the actual video shape.
4. Use the same working benchmark command for every arm. A full-workload prewarm
   batch can be run with `--warmup-requests 0 --num-prompts 5`; discard that batch
   from the measurements. Then run the same command with
   `--warmup-requests 0 --num-prompts 5 --output-file <arm>.json`. These flags
   assume all workload arguments already match your actual request. Preserve your
   original concurrency; use `--max-concurrency 1` when evaluating serial latency.
5. Require all measured requests to complete successfully and return nonempty
   artifacts. Retain response status/bytes, request-to-log correlation, raw timing
   samples, and output hashes separately: the benchmark's aggregate metrics JSON
   alone does not preserve all of that evidence. For an intended lossless change,
   verify output/tensor equivalence and artifact health before accepting latency.
6. Report all three arm means. With equal sample counts, use
   `control = (baseline_before + baseline_after) / 2`,
   `saved_seconds = control - optimized`, and
   `reduction_percent = 100 * saved_seconds / control`.
   Also report baseline drift and sample variation. If the gain is comparable to
   that variation, collect more same-window samples rather than declare a speedup.

An `encode_prompt` improvement alone is insufficient: offload or waiting may
have moved to the next stage. Accept a latency improvement only when the complete
request benefits as well. The synchronized diagnostic pass and the unprofiled
primary pass must each have their own baseline; historical numbers are context,
not the denominator for a new result. Until the target-NPU runs and correctness
checks are complete, label any provisional numbers as exploratory.

## Offline Qwen2.5-Omni and CosyVoice3 examples

Both examples accept `--profiler-config` as a JSON object and forward the parsed
dictionary to `Omni`. Omitting the flag leaves profiling disabled; setting
`VLLM_TORCH_PROFILER_DIR` alone does not enable it. Invalid JSON and non-object
values such as `[]` or `null` are rejected by the argument parser before engine
initialization. Backend-specific fields are validated by the engine configuration.

```bash
python examples/offline_inference/qwen2_5_omni/end2end.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./perf/qwen2_5"}'

python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
  --model /path/to/Fun-CosyVoice3-0.5B \
  --tokenizer /path/to/CosyVoice-BlankEN \
  --ref-audio /path/to/reference.wav \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./perf/cosyvoice3"}'
```

The scripts start profiling before generation and stop it after the requested
outputs have been processed. Qwen2.5-Omni profiles stage 0; CosyVoice3 uses the
engine's default stage selection. For Nsight Systems, use `{"profiler":"cuda"}`
and launch the example under `nsys profile` as described below.

### Shared argument-parser API

Example authors can reuse
[`add_profiler_config_arg`][vllm_omni.utils.profiler.add_profiler_config_arg]:

```python
import argparse

from vllm_omni.utils.profiler import add_profiler_config_arg

parser = argparse.ArgumentParser()
add_profiler_config_arg(parser)
args = parser.parse_args()
# Forward args.profiler_config to Omni(profiler_config=...).
```

The option defaults to `None`. When provided, it uses
[`parse_profiler_config`][vllm_omni.utils.profiler.parse_profiler_config], which
raises `argparse.ArgumentTypeError` for malformed JSON or non-object values.
[`parse_json_object`][vllm_omni.utils.profiler.parse_json_object] provides the same
validation with a caller-supplied flag name. These helpers validate the JSON
object shape only; they do not replace the engine's profiler configuration.

## Diffusion profiling

Diffusion profiling supports two backends through `profiler_config`:

- `torch`: detailed CPU/CUDA traces, operator tables, and optional memory snapshots
- `cuda`: low-overhead CUDA range control for NVIDIA Nsight Systems (`nsys`)

## 1. Configure `profiler_config`

Use `profiler_config` to enable profiling for a diffusion model. For diffusion usage, pass it directly to `Omni(...)` or `vllm serve`.

Minimal torch-profiler config:

```yaml
profiler_config:
  profiler: torch
  torch_profiler_dir: ./perf
```

Supported fields:

| Field | Description |
| --- | --- |
| `profiler` | Profiler backend. Supported values: `torch`, `cuda`. Use `torch` for `trace.json`, Excel operator tables, and optional memory snapshots. Use `cuda` for Nsight Systems only. |
| `torch_profiler_dir` | Output directory for torch-profiler artifacts. Required when `profiler: torch`. |
| `torch_profiler_use_gzip` | Compress `trace_rank*.json` into `trace_rank*.json.gz`. |
| `torch_profiler_record_shapes` | Record input shapes and add a `by_shape` sheet to `ops_rank*.xlsx`. |
| `torch_profiler_with_stack` | Record call stacks, add a `by_stack` sheet to `ops_rank*.xlsx`, and export `stacks_cpu_rank*.txt` and `stacks_cuda_rank*.txt`. |
| `torch_profiler_with_memory` | Enable memory profiling and attempt to dump `memory_snapshot_rank*.pickle`. The pickle is only generated when the current backend supports memory history and snapshot APIs. |
| `torch_profiler_with_flops` | Enable FLOPs collection in `torch.profiler`. This does not add a separate output file. |
| `torch_profiler_dump_cuda_time_total` | Export an additional text summary `profiler_out_<rank>.txt` sorted by `self_cuda_time_total`. |
| `delay_iterations` | Number of worker iterations to skip before profiling starts. |
| `max_iterations` | Maximum number of worker iterations to capture before auto-stop. |
| `wait_iterations` | Torch-profiler wait iterations before warmup. |
| `warmup_iterations` | Torch-profiler warmup iterations. |
| `active_iterations` | Torch-profiler active iterations. |

For detailed explanations of the fields, please refer to upstream vLLM implementation [vllm/config/profiler.py](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/config/profiler.py)

### Minimal configurations by output

Only collect trace output:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with a `summary` sheet

Collect compressed trace output:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_use_gzip": True,
}
```

Outputs:

- `trace_rank*.json.gz`
- `ops_rank*.xlsx` with a `summary` sheet

Collect trace and full operator tables:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with `summary`, `by_shape`, and `by_stack`
- `stacks_cpu_rank*.txt`
- `stacks_cuda_rank*.txt`

Collect trace, operator tables, and memory snapshots:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
    "torch_profiler_with_memory": True,
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with `summary`, `by_shape`, and `by_stack`
- `stacks_cpu_rank*.txt`
- `stacks_cuda_rank*.txt`
- `memory_snapshot_rank*.pickle` when supported by the current backend

### Full torch-profiler configuration

If you want to enable the commonly used torch-profiler options together:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_use_gzip": False,
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
    "torch_profiler_with_memory": True,
    "torch_profiler_with_flops": False,
    "torch_profiler_dump_cuda_time_total": False,
    "delay_iterations": 0,
    "max_iterations": 0,
    "wait_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 1,
}
```

## 2. Profiling Diffusion with PyTorch Profiler

Single-stage diffusion models use `start_profile()` / `stop_profile()` controls. The profiler only writes artifacts after profiling has been started and then stopped.

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./perf",
    },
)

omni.start_profile()
...
omni.stop_profile()
```

For diffusion offline example scripts under `examples/offline_inference/`, pass `--profiler-config` as a JSON object. The script enables profiling when this argument is set and wraps generation with `start_profile()` / `stop_profile()`.

Example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image input.jpg \
  --prompt "A cat playing with yarn" \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_stack": true
  }'
```

Examples:

1. [Image edit example](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)
2. [Image to video example](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)

## 3. Profiling Diffusion with Nsight Systems (`nsys`)

For Nsight Systems, use `profiler: cuda` and wrap the process with `nsys profile`.

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -o diffusion_trace \
  python image_to_video.py ...
```

The Python process being profiled must create the diffusion engine with:

```python
profiler_config = {"profiler": "cuda"}
```

Then call `start_profile()` before the requests you want to capture and `stop_profile()` after them. The diffusion worker processes open and close the CUDA capture range themselves, so `nsys` sees the actual GPU work instead of only the parent process.

## 4. Profiling Online Serving

When `profiler_config.profiler` is set for a diffusion model, the server exposes:

- `POST /start_profile`
- `POST /stop_profile`

> **Note:** `VLLM_TORCH_PROFILER_DIR` was removed in vLLM v0.16.0 when profiling moved to `ProfilerConfig`. A stale value does not enable these endpoints. The server reads it only as a migration diagnostic: if it is set and no stage declares `profiler_config`, it logs a warning at startup and leaves `/start_profile` unregistered. Use `--profiler-config` as shown below, or a `profiler_config` block in the stage YAML. `--profiler-config` is a nested config flag, so it is listed under `vllm serve --help=all` or `vllm serve --help=VllmConfig` rather than in the default `--help` output.

### Start the server

Single-stage diffusion serving with torch profiler:

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --omni \
  --port 8091 \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_profile_wan22_i2v",
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_use_gzip": true,
    "torch_profiler_dump_cuda_time_total": false,
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "delay_iterations": 0,
    "max_iterations": 0,
    "wait_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 1
  }'
```

Single-stage diffusion serving with Nsight Systems:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -o serving_trace \
  vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --omni \
    --port 8091 \
    --profiler-config '{"profiler": "cuda"}'
```

### Control capture

Example profiling flow for an online Qwen-Image request:

```bash
# Start profiling.
curl -X POST http://localhost:8091/start_profile

# Send a Qwen-Image generation request while profiling is active.
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A red vintage bicycle parked beside a quiet canal at sunset"
  }'

# Stop profiling and flush profiler artifacts.
curl -X POST http://localhost:8091/stop_profile
```

## 5. Diffusion Pipeline Profiler

For lightweight per-stage pipeline timing such as `vae.decode` or `diffuse`, see [Diffusion Pipeline Profiler](model/adding_diffusion_model.md#diffusion-pipeline-profiler-performance-profiling). That utility logs stage durations only and does not generate torch-profiler artifacts such as `trace.json`, Excel tables, or memory snapshots.

## 6. Analyze Results

Torch-profiler output:

- Chrome/Perfetto trace: `trace_rank*.json` or `trace_rank*.json.gz`
- Excel workbook: `ops_rank*.xlsx` with `summary`, and optional `by_shape` / `by_stack` sheets
- Stack exports: `stacks_cpu_rank*.txt` and `stacks_cuda_rank*.txt` when stack capture is enabled
- Memory snapshot: `memory_snapshot_rank*.pickle` when memory capture is enabled and supported by the backend
- Optional CUDA-time text summary: `profiler_out_<rank>.txt` when `torch_profiler_dump_cuda_time_total` is enabled

CUDA profiler / Nsight Systems output:

- `.nsys-rep` report files written by `nsys -o ...`

Recommended viewers:

- [Perfetto](https://ui.perfetto.dev/) for torch traces
- `nsys stats <report>.nsys-rep` for CLI summaries
- Nsight Systems GUI for CUDA kernel timelines

For upstream background on the underlying vLLM profiling infrastructure, see the [vLLM profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/).

## 6. Profiling Omni AR/TTS stages

For Omni AR and TTS stages, put `profiler_config` under each stage that should
be captured. The serving API then controls a narrow window with
`POST /start_profile` and `POST /stop_profile`; model initialization and
warmup stay outside the trace.

Example stage override for a host-stack torch trace:

```bash
PROFILE='{"profiler":"torch","torch_profiler_dir":"/tmp/qwen3_tts_profile",'\
'"torch_profiler_with_stack":true,"torch_profiler_record_shapes":false,'\
'"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true}'

vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni \
  --deploy-config vllm_omni/deploy/qwen3_tts_high_concurrency.yaml \
  --stage-overrides "{\"0\":{\"profiler_config\":$PROFILE},\"1\":{\"profiler_config\":$PROFILE}}"
```

After one or more warmup requests, capture one short measured request:

```bash
curl -sS -X POST http://127.0.0.1:8000/start_profile \
  -H 'Content-Type: application/json' -d '{"stages":[0,1]}'
# send one fixed /v1/audio/speech or /v1/chat/completions request here
curl -sS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'Content-Type: application/json' -d '{"stages":[0,1]}'
```

For an Nsight Systems CUDA timeline, use the same stage override with
`{"profiler":"cuda"}` and wrap the server:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree \
  --cpuctxsw=process-tree --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --cuda-graph-trace=node -o /tmp/qwen3_tts_nsys \
  vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni ...
```

For Omni AR workers, `profiler="cuda"` requires the native CUDA profiler
wrapper and uses `cudaProfilerStart/Stop`; the report is only valid if the
`/start_profile` and `/stop_profile` calls both succeed. Use `nsys stats` for
CUDA API, GPU kernel, OS runtime, and NVTX summaries. Treat profiler latency as
diagnostic only; use a separate no-profiler run for performance numbers.

## 7. Orchestrator Monitor

> **Warning:** Diagnostic only. Adds lightweight counters on the orchestrator poll loop and per-replica queue depth. Disable in production unless you are actively debugging orchestrator saturation.

Multi-stage omni pipelines route client-facing outputs and inter-stage connector traffic through a single-process orchestrator. When that loop is saturated (for example by large multimodal payloads on the poll path), TTFT and stability can degrade even when GPU stages are healthy. The orchestrator monitor records time-series signals to confirm whether the orchestrator or downstream stage queues are the bottleneck.

Enable on the stage-0 (orchestrator) process:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --stage-id 0 \
  --enable-orch-monitor
```

Optional output path override:

```bash
export VLLM_OMNI_ORCH_MONITOR_PATH=/tmp/vllm_omni_orch_monitor.json
```

If unset, the monitor writes `./vllm_omni_orch_monitor_<MMDDHHMM>.json` at shutdown.

### Recorded metrics

Each 1-second window records:

| Series | Description |
| --- | --- |
| `windows.duration_s` | Wall time covered by the window |
| `windows.loop_idle` / `windows.loop_active` | Orchestrator poll-loop iterations with no work vs. active forwarding |
| `windows.dispatch_queue_size` | Ready-queue size sampled at each monitor window boundary |
| `windows.dispatch_queue_high_water` | Maximum Event Driven ready-queue size observed in each window |
| `replicas.<stage,replica>.outputs_queue_size` | MP client `outputs_queue` backlog for that replica |
| `replicas.<stage,replica>.inflight` | Requests currently bound/routed to the replica |

On shutdown the server also logs a short summary (`loop_active_pct`, per-replica queue averages/maxima).

`loop_idle` and `loop_active` count loop iterations, so their absolute scale is
tied to which orchestration loop is running. Under the legacy poll loop
(`VLLM_OMNI_EVENT_DRIVEN_ORCH=0`) an idle orchestrator records roughly one
iteration per millisecond. Under the default event-driven loop (see
[Speech API](../serving/speech_api.md#orchestration-loop-experimental)) an idle
orchestrator wakes only once per 0.5 s reconcile timeout, while a busy one still
records one iteration per routed output. Window counts and the `loop_active_pct`
summary are therefore not comparable across the two modes.

For Event Driven runs, use `dispatch_queue_high_water` rather than only the
window-boundary `dispatch_queue_size`: a serial dispatcher can drain a burst
before the window closes, hiding a transient ready-queue backlog in the latter.

### Relationship to other diagnostics

This monitor is intentionally separate from the existing profiling tools:

| Tool | Scope | Output |
| --- | --- | --- |
| `--enable-diffusion-pipeline-profiler` | Diffusion stage functions (`vae.decode`, `diffuse`, …) | Per-request `stage_durations` / logs |
| `--enable-ar-profiler` | AR stage generation time | Per-request `stage_durations` |
| `profiler_config` (`torch` / `cuda`) | GPU/CPU kernels inside a stage worker | `trace.json`, Nsight, operator tables |
| Prometheus `/metrics` (`vllm:omni_*`) | Pipeline SLOs and cross-stage transfer | Continuous scrape endpoint |
| `--enable-orch-monitor` | Orchestrator poll loop + replica queue backlog | Single JSON file at shutdown |

The orchestrator monitor does not use `torch.profiler` because the bottleneck signal lives in the orchestrator process (poll-loop duty cycle and queue depth), not inside stage workers. It complements Prometheus metrics when you need a post-run time series for a benchmark or regression run.

Process liveness monitors (`monitor_engine_liveness`, worker/engine-core monitors) are unrelated: they detect crashed workers, not orchestrator load.
