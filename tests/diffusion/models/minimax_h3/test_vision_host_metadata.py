# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

POS_TENSOR = "VLLM_OMNI_H3_VISION_POS_TENSOR"
POS_REUSE = "VLLM_OMNI_H3_VISION_POS_REUSE"
LENGTHS = "VLLM_OMNI_H3_VISION_LENGTHS"
POS_CACHE = "VLLM_OMNI_H3_VISION_POS_CACHE"


@pytest.fixture(autouse=True)
def disable_metadata_candidates(monkeypatch):
    for key in (POS_TENSOR, POS_REUSE, LENGTHS, POS_CACHE):
        monkeypatch.delenv(key, raising=False)


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


def make_pixels(model, grid):
    count = sum(t * h * w for t, h, w in grid.tolist())
    width = model.config.in_channels * model.config.temporal_patch_size * model.config.patch_size**2
    generator = torch.Generator().manual_seed(123)
    return torch.randn(count, width, generator=generator).to(model.pos_embed.weight.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_position_cache_hits_and_keeps_weight_updates_live(monkeypatch, dtype):
    model = make_model(dtype)
    grid = torch.tensor([[1, 4, 6], [1, 4, 6], [1, 6, 8]])
    with torch.inference_mode():
        expected = model.fast_pos_embed_interpolate(grid)
        monkeypatch.setenv(POS_CACHE, "1")
        assert_bytes(model.fast_pos_embed_interpolate(grid), expected)

        def no_geometry(*args, **kwargs):
            raise AssertionError("cache hit rebuilt geometry")

        monkeypatch.setattr(torch, "linspace", no_geometry)
        assert_bytes(model.fast_pos_embed_interpolate(grid), expected)
        model.pos_embed.weight.add_(1)
        actual = model.fast_pos_embed_interpolate(grid)
        assert not torch.equal(actual, expected)


def test_position_cache_is_bounded_and_separates_order_and_dtype(monkeypatch):
    model = make_model()
    monkeypatch.setenv(POS_CACHE, "1")
    with torch.inference_mode():
        for h in range(2, 22, 2):
            model.fast_pos_embed_interpolate(torch.tensor([[1, h, 4]]))
        assert len(model._position_metadata_cache) <= 8
        assert model._position_metadata_cache.nbytes <= 8 * 1024**2
        for rows in ([[1, 4, 6], [1, 6, 8]], [[1, 6, 8], [1, 4, 6]]):
            grid = torch.tensor(rows)
            actual = model.fast_pos_embed_interpolate(grid)
            monkeypatch.setenv(POS_CACHE, "0")
            assert_bytes(actual, model.fast_pos_embed_interpolate(grid))
            monkeypatch.setenv(POS_CACHE, "1")
        model.to(dtype=torch.bfloat16)
        assert len(model._position_metadata_cache) == 0
        assert model.fast_pos_embed_interpolate(grid).dtype == torch.bfloat16


def test_cpu_metadata_cache_rejects_oversized_entries():
    from vllm_omni.diffusion.models.minimax_h3.vision_metadata import CpuTensorCache

    cache = CpuTensorCache(max_entries=2, max_bytes=16)
    cache.put("a", (torch.ones(2),))
    cache.put("large", (torch.ones(5),))
    assert cache.get("a") is not None
    assert cache.get("large") is None
    cache.put("b", (torch.zeros(2),))
    cache.get("a")
    cache.put("c", (torch.zeros(2),))
    assert cache.get("b") is None
    assert cache.nbytes == 16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("rows", [[[1, 4, 6]], [[1, 4, 6], [1, 6, 8]], [[2, 4, 6], [1, 6, 8]]])
def test_shared_lengths_preserve_full_vision_and_deepstack(monkeypatch, dtype, combined, rows):
    model = make_model(dtype)
    grid = torch.tensor(rows)
    pixels = make_pixels(model, grid)
    with torch.inference_mode():
        expected = model(pixels, grid)
        monkeypatch.setenv(LENGTHS, "1")
        if combined:
            monkeypatch.setenv(POS_TENSOR, "1")
            monkeypatch.setenv(POS_REUSE, "1")
        actual = model(pixels, grid)
    assert_bytes(actual[0], expected[0])
    assert len(actual[1]) == len(expected[1])
    for a, b in zip(actual[1], expected[1]):
        assert_bytes(a, b)


@pytest.mark.parametrize("depth", [3, 27])
@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_shared_lengths_read_once_for_all_layers_and_keep_hooks(monkeypatch, depth, enabled):
    model = make_model(depth=depth)
    grid = torch.tensor([[2, 4, 6], [1, 6, 8]])
    pixels = make_pixels(model, grid)
    if enabled is not None:
        monkeypatch.setenv(LENGTHS, enabled)
    original = torch.Tensor.tolist
    reads = []
    seen = []

    def tracked(tensor):
        if tensor.dtype == torch.int32 and tuple(tensor.shape) == (3,):
            reads.append(True)
        return original(tensor)

    def observe(module, args, kwargs):
        seen.append(kwargs.get("sequence_lengths"))

    handles = [block.register_forward_pre_hook(observe, with_kwargs=True) for block in model.blocks]
    monkeypatch.setattr(torch.Tensor, "tolist", tracked)
    try:
        with torch.inference_mode():
            model(pixels, grid)
    finally:
        for handle in handles:
            handle.remove()
    assert len(reads) == (1 if enabled == "1" else depth * 3)
    assert len(seen) == depth
    if enabled == "1":
        assert all(value is seen[0] for value in seen)
        assert seen[0] == (24, 24, 48)
    else:
        assert all(value is None for value in seen)


def test_shared_lengths_are_rebuilt_for_different_requests(monkeypatch):
    model = make_model()
    for rows in ([[1, 4, 6]], [[2, 8, 4], [1, 6, 8]], [[1, 4, 6]]):
        grid = torch.tensor(rows)
        pixels = make_pixels(model, grid)
        with torch.inference_mode():
            monkeypatch.setenv(LENGTHS, "0")
            expected = model(pixels, grid)
            monkeypatch.setenv(LENGTHS, "1")
            actual = model(pixels, grid)
        assert_bytes(actual[0], expected[0])
        for a, b in zip(actual[1], expected[1]):
            assert_bytes(a, b)
