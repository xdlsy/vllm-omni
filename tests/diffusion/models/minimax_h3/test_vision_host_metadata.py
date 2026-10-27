# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

POS_TENSOR = "VLLM_OMNI_H3_VISION_POS_TENSOR"
POS_REUSE = "VLLM_OMNI_H3_VISION_POS_REUSE"


def make_model(dtype=torch.float32, depth=3):
    config = SimpleNamespace(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        spatial_merge_size=2,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=16,
        depth=depth,
        out_hidden_size=24,
        deepstack_visual_indexes=[i for i in (0, 1, 8, 16, 24) if i < depth],
    )
    torch.manual_seed(42)
    return encoder_module.MiniMaxH3Qwen3VLVisionModel(config).to(dtype=dtype).eval()


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [[[1, 4, 6]], [[1, 4, 6], [1, 4, 6], [1, 6, 8]], [[2, 4, 6], [3, 8, 4]]])
def test_tensor_position_metadata_matches_legacy(monkeypatch, dtype, rows):
    model = make_model(dtype)
    grid = torch.tensor(rows)
    monkeypatch.delenv(POS_TENSOR, raising=False)
    with torch.inference_mode():
        expected = model.fast_pos_embed_interpolate(grid)
        monkeypatch.setenv(POS_TENSOR, "1")
        actual = model.fast_pos_embed_interpolate(grid)
    assert_bytes(actual, expected)


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_tensor_position_metadata_avoids_scalar_lists(monkeypatch, enabled):
    model = make_model()
    grid = torch.tensor([[1, 4, 6], [1, 6, 8]])
    if enabled is None:
        monkeypatch.delenv(POS_TENSOR, raising=False)
    else:
        monkeypatch.setenv(POS_TENSOR, enabled)
    original = torch.Tensor.tolist
    calls = []

    def tracked(tensor):
        calls.append(tuple(tensor.shape))
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", tracked)
    with torch.inference_mode():
        model.fast_pos_embed_interpolate(grid)
    assert len(calls) == (1 if enabled == "1" else 17)


def test_tensor_position_metadata_does_not_retain_previous_grid(monkeypatch):
    model = make_model()
    for rows in ([[1, 4, 6]], [[2, 8, 4], [1, 6, 8]], [[1, 4, 6]]):
        grid = torch.tensor(rows)
        with torch.inference_mode():
            monkeypatch.setenv(POS_TENSOR, "0")
            expected = model.fast_pos_embed_interpolate(grid)
            monkeypatch.setenv(POS_TENSOR, "1")
            actual = model.fast_pos_embed_interpolate(grid)
        assert_bytes(actual, expected)


@pytest.mark.parametrize("tensor_path", ["0", "1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [[[1, 4, 6]], [[1, 4, 6], [1, 4, 6], [1, 6, 8]], [[2, 4, 6], [3, 4, 6]]])
def test_reused_position_metadata_matches_uncached(monkeypatch, tensor_path, dtype, rows):
    model = make_model(dtype)
    grid = torch.tensor(rows)
    monkeypatch.setenv(POS_TENSOR, tensor_path)
    with torch.inference_mode():
        monkeypatch.setenv(POS_REUSE, "0")
        expected = model.fast_pos_embed_interpolate(grid)
        monkeypatch.setenv(POS_REUSE, "1")
        actual = model.fast_pos_embed_interpolate(grid)
    assert_bytes(actual, expected)


@pytest.mark.parametrize("tensor_path", ["0", "1"])
@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_reused_position_metadata_computes_each_size_once(monkeypatch, tensor_path, enabled):
    model = make_model()
    grid = torch.tensor([[1, 4, 6], [1, 4, 6], [2, 6, 8], [1, 8, 4], [1, 6, 8], [3, 4, 6]])
    monkeypatch.setenv(POS_TENSOR, tensor_path)
    if enabled is None:
        monkeypatch.delenv(POS_REUSE, raising=False)
    else:
        monkeypatch.setenv(POS_REUSE, enabled)
    original = torch.linspace
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "linspace", tracked)
    with torch.inference_mode():
        model.fast_pos_embed_interpolate(grid)
    assert len(calls) == (6 if enabled == "1" else 12)


@pytest.mark.parametrize("tensor_path", ["0", "1"])
def test_reused_position_metadata_is_request_local(monkeypatch, tensor_path):
    model = make_model()
    monkeypatch.setenv(POS_TENSOR, tensor_path)
    monkeypatch.setenv(POS_REUSE, "1")
    original = torch.linspace
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "linspace", tracked)
    with torch.inference_mode():
        model.fast_pos_embed_interpolate(torch.tensor([[1, 4, 6], [2, 4, 6]]))
        model.fast_pos_embed_interpolate(torch.tensor([[1, 4, 6], [1, 6, 8]]))
        model.fast_pos_embed_interpolate(torch.tensor([[1, 4, 6], [2, 4, 6]]))
    assert len(calls) == 8
