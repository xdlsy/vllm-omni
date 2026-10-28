# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.minimax_h3.test_vision_host_metadata import assert_bytes, make_model, make_pixels
from vllm_omni.diffusion.models.minimax_h3 import vision_metadata

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]
ROTARY_CPU = "VLLM_OMNI_H3_VISION_ROTARY_CPU"


@pytest.fixture(autouse=True)
def disable_candidates(monkeypatch):
    for suffix in (
        "POS_TENSOR",
        "POS_REUSE",
        "LENGTHS",
        "POS_CACHE",
        "ROTARY_CPU",
        "EARLY_PREFETCH",
        "RESIDENT_FIRST",
        "WORKSPACE",
    ):
        monkeypatch.delenv("VLLM_OMNI_H3_VISION_" + suffix, raising=False)


@pytest.mark.parametrize("rows", [[[1, 4, 6]], [[2, 4, 6], [1, 8, 4]], [[1, 4, 6], [1, 4, 6]]])
def test_cpu_rotary_coordinates_follow_merge_and_frame_order(rows):
    expected = [
        [r * 2 + i, c * 2 + j]
        for t, h, w in rows
        for _ in range(t)
        for r in range(h // 2)
        for c in range(w // 2)
        for i in range(2)
        for j in range(2)
    ]
    actual = vision_metadata.build_rotary_positions(rows, 2)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.long
    assert actual.tolist() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_rotary_path_preserves_full_vision(monkeypatch, dtype):
    model = make_model(dtype)
    grid = torch.tensor([[2, 4, 6], [1, 6, 8]])
    pixels = make_pixels(model, grid)
    calls = []
    original = vision_metadata.build_rotary_positions

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(vision_metadata, "build_rotary_positions", tracked)
    with torch.inference_mode():
        expected = model(pixels, grid)
        monkeypatch.setenv(ROTARY_CPU, "1")
        actual = model(pixels, grid)
    assert len(calls) == 1
    assert_bytes(actual[0], expected[0])
    for a, b in zip(actual[1], expected[1]):
        assert_bytes(a, b)


def test_early_prefetch_precedes_patch_and_keeps_block_calls(monkeypatch):
    model = make_model()
    order = []
    hook = SimpleNamespace(prefetch_current_layer=lambda: order.append("prefetch"))
    model.blocks[0]._hook_registry = SimpleNamespace(get_hook=lambda name: hook)
    handle = model.patch_embed.register_forward_pre_hook(lambda *_: order.append("patch"))
    grid = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        model(make_pixels(model, grid), grid)
        assert order == ["patch"]
        order.clear()
        monkeypatch.setenv("VLLM_OMNI_H3_VISION_EARLY_PREFETCH", "1")
        model(make_pixels(model, grid), grid)
    handle.remove()
    assert order == ["prefetch", "patch"]
