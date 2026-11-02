# MiniMax H3 early non-block staging (experimental)

`VLLM_OMNI_H3_NONBLOCK_PREFETCH=1` starts manual layerwise-offloaded encoder component residency before CPU image preprocessing. The default is `0`. It reuses the existing non-block stager, copy stream and ready event; it does not preload all transformer blocks or retain weights between requests.

Participating encoder ranks acquire one request-local component scope. A success flag is reduced on the existing world CPU/Gloo group before preparation/device collectives. On failure, ranks gather error metadata and release acquired scopes. A device-group status collective must not be substituted: it could wait behind the stager ready dependency and remove the intended overlap.

An already-staged marker prevents a nested load/offload scope at text encoding. The outer scope releases residency after result broadcast, including on exceptions. Non-encoder ranks do not stage; resident and model-level sequential-offload paths retain their original execution route. Enabling the experimental flag still adds CPU status coordination, so unsupported configurations should leave it disabled.

Measure the complete encode_prompt interval, including early allocation/status coordination and final release. The internal text-encoding interval changes boundaries under this option and is not an independent speedup metric. Additional residency duration, peak memory and host-memory bandwidth contention must be checked on the actual workload. Event readiness observations establish completion ordering, not an exact DMA overlap duration.

Validate with identical deterministic settings for OFF and ON, complete condition hidden/tags and final media bytes, actual nonzero-rank load-failure cleanup, and same-window performance. Image preprocessing and hidden-device flags are separate candidates and should be held fixed during attribution runs. This option remains default-off; no benefit is implied by enabling it.
