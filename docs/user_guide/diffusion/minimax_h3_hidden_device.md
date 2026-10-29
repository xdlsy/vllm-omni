# MiniMax H3 hidden output placement (experimental)

`VLLM_OMNI_H3_HIDDEN_DEVICE=1` lets the pipeline retain the encoder hidden tensor on its activation device until the existing result broadcast. The default is `0`.

The regular `encode_ids()` and module `forward()` CPU-return behavior is preserved. Their keyword-only `return_on_device=False` option can be enabled by the internal pipeline path. This changes output placement only: encoder computation, dtype, collective order, validation, SDPA restoration and component offload remain unchanged. The sequential-offload branch still invokes the module through `nn.Module.__call__` so hooks execute.

With text TP8, the previous path returns CPU hidden on all eight encoder ranks, then moves rank0's source back to the device for broadcast. Retaining the activation can avoid those transfers. The activation must remain valid after component offload; parameter storage and activation storage have separate ownership.

Use the same deterministic configuration for OFF and ON validation. On the tested Ascend environment, default FP32 TP AllReduce repetitions can differ; a fixed seed alone does not establish bitwise repeatability. Check complete hidden/tags and final media bytes before performance timing. Verify the flag's output-device effect and component cleanup separately. This option remains experimental until the workload-specific precision and same-window performance gates pass.

This flag does not enable image preprocessing optimization, retain model weights between requests, cache encoded conditions, remove stream waits, or change the standalone encoder's default output device.
