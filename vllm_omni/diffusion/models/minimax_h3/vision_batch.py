# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Equal-length image attention batching for isolated correctness experiments."""

import torch


def run_grouped_attention(qs, ks, vs, *, attention):
    """Pack images on batch axis, retain their independent token sequences."""
    if not (len(qs) == len(ks) == len(vs)):
        raise ValueError("segment lists must match")
    groups = {}
    for index, (q, k, v) in enumerate(zip(qs, ks, vs)):
        if q.ndim != 4 or q.shape[0] != 1 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("expected matching single-image BHSD tensors")
        groups.setdefault(tuple(q.shape), []).append(index)
    outputs: dict[int, torch.Tensor] = {}
    for indices in groups.values():
        if len(indices) == 1:
            index = indices[0]
            outputs[index] = attention(qs[index], ks[index], vs[index])
            continue
        inputs = [torch.cat([parts[i] for i in indices], dim=0) for parts in (qs, ks, vs)]
        result = attention(*inputs)
        for row, index in enumerate(indices):
            outputs[index] = result[row : row + 1]
    return [outputs[index] for index in range(len(qs))]
