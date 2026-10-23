# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Group packing contracts; NPU bitwise gates remain separate."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_batch_isolates_images_and_restores_order():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_batch.py"
    spec = importlib.util.spec_from_file_location("batch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    segments = [torch.full((1, 2, n, 3), float(i)) for i, n in enumerate((8, 8, 4, 6, 6, 12))]
    calls = []

    def attention(q, k, v):
        calls.append(q.shape)
        if q.shape[0] == 1:
            # Singleton groups must retain the original view/stride, avoiding copies.
            assert any(q is original for original in segments)
        return v + q.mean(dim=2, keepdim=True)

    result = module.run_grouped_attention(segments, segments, segments, attention=attention)
    assert len(calls) == 4
    assert [s[0] for s in calls] == [2, 1, 2, 1]
    for i, output in enumerate(result):
        assert torch.equal(output, segments[i] * 2)
