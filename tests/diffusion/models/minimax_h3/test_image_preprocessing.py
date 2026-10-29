# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from vllm_omni.diffusion.models.minimax_h3 import image_preprocessing as candidate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_H3_IMAGE_PREPROCESS", "1")
    monkeypatch.delenv("VLLM_OMNI_H3_IMAGE_PREPROCESS_VERIFY", raising=False)


def images(shapes):
    rng = np.random.default_rng(42)
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in shapes]


def processor(**kwargs):
    return Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 32 * 32, "longest_edge": 1024 * 1024}, **kwargs)


def assert_same(actual, expected):
    for key in ("pixel_values", "image_grid_thw"):
        ref = expected[key].to(actual[key].dtype)
        assert actual[key].is_contiguous()
        assert actual[key].shape == ref.shape
        assert torch.equal(actual[key].view(torch.uint8), ref.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 96)],
        [(65, 99), (32, 64), (65, 99)],
        [(864, 480), (864, 480), (640, 352), (480, 352), (480, 352), (480, 864)],
    ],
)
@pytest.mark.parametrize("grouping", [None, True, False])
def test_real_processor_byte_parity(shapes, grouping):
    p = processor(disable_grouping=grouping)
    batch = images(shapes)
    before = [np.array(image) for image in batch]
    actual = candidate.preprocess_images(p, batch)
    assert actual["pixel_values"].dtype == torch.bfloat16
    assert_same(actual, p(images=batch, return_tensors="pt"))
    assert all(np.array_equal(old, np.array(image)) for old, image in zip(before, batch))
    assert "_preprocess" not in p.__dict__


@pytest.mark.parametrize(
    "options",
    [
        {"temporal_patch_size": 1},
        {"temporal_patch_size": 3},
        {"merge_size": 1},
        {"do_rescale": False},
        {"do_normalize": False},
        {"do_normalize": False, "do_rescale": False},
        {"rescale_factor": 0.003, "image_mean": [0.2, 0.3, 0.4], "image_std": [0.5, 0.6, 0.7]},
    ],
)
def test_processor_options(options):
    p = processor(**options)
    batch = images([(64, 96), (96, 64)])
    assert_same(candidate.preprocess_images(p, batch), p(images=batch, return_tensors="pt"))


def test_all_uint8_values_and_grayscale():
    batch = [Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (32, 1))), *images([(64, 96)])]
    p = processor()
    assert_same(candidate.preprocess_images(p, batch), p(images=batch, return_tensors="pt"))


@pytest.mark.parametrize("enabled", [None, "0"])
def test_default_path_and_custom_processor(monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("VLLM_OMNI_H3_IMAGE_PREPROCESS")
    else:
        monkeypatch.setenv("VLLM_OMNI_H3_IMAGE_PREPROCESS", enabled)
    p = processor()
    batch = images([(64, 96)])
    actual = candidate.preprocess_images(p, batch)
    assert actual["pixel_values"].dtype == torch.float32
    assert_same(actual, p(images=batch, return_tensors="pt"))


def test_unknown_and_overridden_processor_fall_back():
    class CustomProcessor:
        def __call__(self, **kwargs):
            return {"custom": kwargs}

    batch = images([(64, 96)])
    assert candidate.preprocess_images(CustomProcessor(), batch)["custom"]["images"] is batch

    class Subclass(Qwen2VLImageProcessor):
        def _preprocess(self, images, **kwargs):
            return {"custom": True}

    assert candidate.preprocess_images(Subclass(), batch) == {"custom": True}


def test_no_final_cat_and_request_owned_storage(monkeypatch):
    p = processor()
    batch = images([(64, 96), (96, 64)])
    expected = p(images=batch, return_tensors="pt")

    def forbidden(*args, **kwargs):
        pytest.fail("fast packing must not concatenate per-image patch buffers")

    monkeypatch.setattr(torch, "cat", forbidden)
    first = candidate.preprocess_images(p, batch)
    snapshot = first["pixel_values"].clone()
    second = candidate.preprocess_images(p, images([(32, 64)]))
    assert first["pixel_values"].data_ptr() != second["pixel_values"].data_ptr()
    assert torch.equal(snapshot, first["pixel_values"])
    assert_same(first, expected)


def test_shared_processor_concurrent_requests():
    p = processor()
    batches = [images([(64, 96)]), images([(96, 64), (32, 64)])]
    expected = [p(images=batch, return_tensors="pt") for batch in batches]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda batch: candidate.preprocess_images(p, batch), batches))
    for result, reference in zip(results, expected):
        assert_same(result, reference)


def test_verification_detects_bad_packing(monkeypatch):
    p = processor()
    monkeypatch.setenv("VLLM_OMNI_H3_IMAGE_PREPROCESS_VERIFY", "1")
    batch = images([(64, 96)])
    assert_same(candidate.preprocess_images(p, batch), p(images=batch, return_tensors="pt"))
    original = candidate._packed_preprocess

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["pixel_values"][0, 0] += 1
        return result

    monkeypatch.setattr(candidate, "_packed_preprocess", corrupt)
    with pytest.raises(AssertionError, match="pixel_values"):
        candidate.preprocess_images(p, batch)


def test_instance_method_override_is_preserved():
    from types import MethodType

    p = processor()
    p._preprocess = MethodType(lambda self, images, **kwargs: {"override": True}, p)
    assert candidate.preprocess_images(p, images([(64, 96)])) == {"override": True}


def test_legacy_backend_import_falls_back(monkeypatch):
    import builtins

    p = processor()
    batch = images([(64, 96)])
    reference = p(images=batch, return_tensors="pt")
    original = builtins.__import__

    def legacy(name, *args, **kwargs):
        if name == "transformers.image_processing_backends":
            raise ImportError("older Transformers has no TorchvisionBackend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", legacy)
    actual = candidate.preprocess_images(p, batch)
    assert actual["pixel_values"].dtype == torch.float32
    assert_same(actual, reference)


@pytest.mark.parametrize("patch,merge", [(8, 2), (16, 1), (16, 4)])
def test_other_patch_geometry_and_no_resize(patch, merge):
    p = Qwen2VLImageProcessor(patch_size=patch, merge_size=merge, do_resize=False)
    batch = images([(64, 128), (128, 64)])
    assert_same(candidate.preprocess_images(p, batch), p(images=batch, return_tensors="pt"))


@pytest.mark.parametrize(
    "size", [{"shortest_edge": 16384, "longest_edge": 32768}, {"shortest_edge": 1024, "longest_edge": 4096}]
)
def test_resize_pixel_limits(size):
    p = Qwen2VLImageProcessor(patch_size=16, size=size)
    batch = images([(32, 64), (256, 128)])
    assert_same(candidate.preprocess_images(p, batch), p(images=batch, return_tensors="pt"))


def test_non_cpu_prepared_tensor_delegates():
    calls = []
    marker = object()
    image = torch.empty((3, 64, 64), device="meta")

    def reference(**kwargs):
        calls.append(kwargs)
        return marker

    assert candidate._packed_preprocess(None, [image], reference=reference) is marker
    assert calls[0]["images"][0] is image


def test_new_resize_factor_contract(monkeypatch):
    from types import MethodType

    from transformers.image_processing_backends import TorchvisionBackend
    from transformers.image_utils import SizeDict
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    p = processor()
    batch = images([(65, 99), (96, 64)])
    expected = p(images=batch, return_tensors="pt")
    calls = []

    def modern_resize(self, images, size, resample, factor):
        calls.append(factor)
        height, width = smart_resize(
            *images.shape[-2:], factor=factor, min_pixels=size.shortest_edge, max_pixels=size.longest_edge
        )
        return TorchvisionBackend.resize(self, images, size=SizeDict(height=height, width=width), resample=resample)

    monkeypatch.setattr(p, "resize", MethodType(modern_resize, p))
    assert_same(candidate.preprocess_images(p, batch), expected)
    assert calls and all(factor == 32 for factor in calls)


def test_pipeline_image_call_uses_candidate():
    """Exercise the production method without importing the unavailable vLLM runtime."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    source = Path(candidate.__file__).with_name("pipeline_minimax_h3.py")
    tree = ast.parse(source.read_text(encoding="utf8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "encode_prompt")
    namespace = {
        "torch": torch,
        "preprocess_images": candidate.preprocess_images,
        "_dit_rank_world": lambda: (None, 0, 1),
        "_broadcast_rank0_exception": lambda error: (_ for _ in ()).throw(error) if error else None,
        "_broadcast_tensor": lambda value, **kwargs: value,
        "build_minimax_h3_presentation": lambda *args, **kwargs: (torch.tensor([1]), torch.tensor([2])),
    }
    prefix = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=prefix + [method], type_ignores=[]), str(source), "exec"), namespace)
    captured = {}

    def distribute(ids, vision):
        captured.update(vision)
        return ids

    owner = SimpleNamespace(
        processor=SimpleNamespace(image_processor=processor()),
        tokenizer=None,
        text_encoder_tp_size=1,
        device="cpu",
        _distribute_encode_inputs=distribute,
        _encode_text_hidden=lambda *args: torch.zeros(1, 4),
    )
    prepared = SimpleNamespace(
        images=images([(64, 96)]),
        qwen_videos=[],
        prompt="x",
        media=SimpleNamespace(task="ref2va"),
        condition_labels=[],
        video_timestamps=[],
    )
    namespace["encode_prompt"](owner, prepared)
    assert captured["pixel_values"].dtype == torch.bfloat16
