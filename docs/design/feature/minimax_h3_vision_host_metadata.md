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
