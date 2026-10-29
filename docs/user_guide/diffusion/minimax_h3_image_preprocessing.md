# Experimental MiniMax-H3 CPU image packing

This private experiment targets reference-image preparation before the vision
encoder. It is disabled by default and does not change video processing.

Set `VLLM_OMNI_H3_IMAGE_PREPROCESS=1` to enable request-owned BF16 patch packing.
The supported backend is the stock Qwen2VL Torchvision image processor. Its
public input conversion, RGB conversion, resize, rescale and normalization are
retained. The candidate copies the strided patch layout directly into one BF16
output buffer, eliminating per-image flattened FP32 patch buffers, their final
concatenation, and the subsequent standalone CPU BF16 conversion. The downstream
vision input is still contiguous BF16; `image_grid_thw` retains its original
integer values and order.

Unknown backends, subclasses, instance-overridden preprocessing methods, and
older Transformers without the supported Torchvision backend use the original
processor. Unsupported execution is not evidence that the optimization ran.
Non-CPU tensors also retain the original preprocessing path. No cross-request
image cache or mutable shared workspace is introduced.

For correctness validation, also set
`VLLM_OMNI_H3_IMAGE_PREPROCESS_VERIFY=1`. Every invocation then runs the original
processor and compares the final BF16 pixel bytes and grid bytes. A mismatch
raises an assertion; the extra work is intentional and must be disabled for
timing. Unset the enable flag, or set it to `0`, to restore the original path.

Local checks use real CPU torch, Torchvision and Transformers with synthetic
RGB inputs, including the historical six-image geometry. They establish
component-level byte parity and allocation changes only. They do not establish
NPU model correctness, media determinism or end-to-end performance. Before
adoption, validate the exact processor revision in the serving image, compare
vision inputs and outputs, and measure the full request from a common start.
