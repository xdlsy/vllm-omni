# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU storage/lifecycle checks; accelerator ordering requires hardware tests."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.encoder_residency import EncoderResidency, EncoderResidencyOptions

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def cpu_staging_runtime(monkeypatch):
    """Keep CPU lifecycle tests independent of accelerator runtime support."""
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm_omni.diffusion.offloader.module_residency import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "Stream", Mock)
    monkeypatch.setattr(current_omni_platform, "Event", Mock)
    monkeypatch.setattr(current_omni_platform, "current_stream", Mock)
    monkeypatch.setattr(current_omni_platform, "stream", lambda _: nullcontext())
    monkeypatch.setattr(current_omni_platform, "synchronize", Mock())
    monkeypatch.setattr(current_omni_platform, "empty_cache", Mock())


def model():
    result = nn.Module()
    result.vision = nn.Sequential(nn.Linear(3, 3))
    result.text_model = nn.Module()
    result.text_model.embed_tokens = nn.Linear(3, 3)
    result.text_model.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
    return result


def test_frozen_master_rebinds_without_d2h():
    module = model()
    expected = module.text_model.layers[0].weight.detach().clone()
    options = EncoderResidencyOptions(frozen_master=True)
    residency = EncoderResidency(module, torch.device("cpu"), options)
    master = module.text_model.layers[0].weight.data_ptr()
    for _ in range(3):
        residency.load()
        assert module.text_model.layers[0].weight.data_ptr() != master
        residency.offload()
        assert module.text_model.layers[0].weight.data_ptr() == master
        assert torch.equal(module.text_model.layers[0].weight, expected)
    assert residency.stats["loads"] == 3
    assert residency.stats["offloads"] == 3


def test_default_off_and_invalid_environment():
    assert not EncoderResidencyOptions.from_env({}).enabled
    with pytest.raises(ValueError, match="0 or 1"):
        EncoderResidencyOptions.from_env({"VLLM_OMNI_ENCODER_FROZEN_MASTER": "yes"})


def lifecycle_class(relative_path, class_name, methods, namespace):
    """Execute unchanged lifecycle methods without importing model kernels."""
    source = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    namespace["nn"] = nn
    definition.bases = [ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())]
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition]
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace[class_name]


def test_full_encoder_and_sequential_hook_use_the_same_controller():
    from vllm_omni.diffusion.offloader.module_residency import current_omni_platform

    encoder_type = lifecycle_class(
        "vllm_omni/diffusion/models/minimax_h3/encoder.py",
        "MiniMaxH3Qwen3VLEncoder",
        {
            "is_loaded",
            "uses_omni_encoder_residency",
            "_encoder_residency_controller",
            "load_to_device",
            "offload_to_cpu",
        },
        dict(EncoderResidency=EncoderResidency, EncoderResidencyOptions=EncoderResidencyOptions, torch=torch),
    )
    hook_type = lifecycle_class(
        "vllm_omni/diffusion/offloader/sequential_backend.py",
        "SequentialOffloadHook",
        {"_uses_encoder_residency", "_to_cpu", "_to_gpu"},
        dict(current_omni_platform=current_omni_platform),
    )
    encoder = encoder_type()
    owner = model()
    encoder.vision, encoder.text_model = owner.vision, owner.text_model
    encoder.quant_config = None
    encoder.device_target = torch.device("cpu")
    encoder._encoder_residency_options = EncoderResidencyOptions(frozen_master=True)
    # Controller snapshots need nn.Module traversal, forwarded by this lightweight owner.
    encoder.parameters, encoder.buffers = owner.parameters, owner.buffers
    hook = hook_type()
    hook.use_hsdp = False
    hook._to_gpu(encoder)
    controller = encoder._encoder_residency
    hook._to_cpu(encoder)
    assert controller.stats == {"loads": 1, "offloads": 1}
    hook.use_hsdp = True
    with pytest.raises(ValueError, match="HSDP"):
        hook._to_gpu(encoder)
    encoder._omni_layerwise_enabled = True
    with pytest.raises(ValueError, match="legacy layerwise"):
        encoder.load_to_device()


def test_cache_retention_is_independent_and_obeys_bounds():
    from vllm_omni.diffusion.offloader.module_residency import current_omni_platform

    options = EncoderResidencyOptions.from_env({"VLLM_OMNI_ENCODER_CACHE_RETENTION": "1"})
    residency = EncoderResidency(model(), torch.device("cpu"), options)
    assert not residency.stagers  # cache-only leaves the legacy copies intact
    with (
        patch.object(torch.accelerator, "memory_reserved", return_value=20),
        patch.object(torch.accelerator, "memory_allocated", return_value=10),
        patch.object(current_omni_platform, "get_device_memory", return_value=(80, 100)),
        patch.object(current_omni_platform, "empty_cache") as release,
    ):
        residency.load()
        residency.offload()
        release.assert_not_called()
    with (
        patch.object(residency.cache, "_should_release", return_value=True),
        patch.object(current_omni_platform, "empty_cache") as release,
    ):
        residency.load()
        residency.offload()
        release.assert_called_once()


def test_pinned_masters_are_allocated_once_and_reused():
    calls = []

    def pin(tensor):
        calls.append(tensor.numel())
        return tensor.clone()

    options = EncoderResidencyOptions.from_env({"VLLM_OMNI_ENCODER_PINNED_BUFFERS": "1"})
    with patch.object(torch.Tensor, "pin_memory", pin):
        residency = EncoderResidency(model(), torch.device("cpu"), options)
        assert calls
        initial_calls = len(calls)
        masters = [group.master.data_ptr() for group in residency.stagers[0]._groups]
        for _ in range(3):
            residency.load()
            residency.offload()
        assert len(calls) == initial_calls
        assert masters == [group.master.data_ptr() for group in residency.stagers[0]._groups]


@pytest.mark.parametrize("resident_layers", [-1, 2])
def test_full_and_partial_device_residency_reuse_weights(resident_layers):
    owner = model()
    residency = EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(resident_layers=resident_layers))
    residency.load()
    retained = [stager for stager in residency.stagers if stager.retain_device_storage]
    assert len(retained) == 1
    ptrs = [storage.data_ptr() for storage in retained[0]._device_storages]
    residency.offload()
    assert ptrs == [storage.data_ptr() for storage in retained[0]._device_storages]
    residency.load()
    assert ptrs == [storage.data_ptr() for storage in retained[0]._device_storages]


def test_partition_rejects_shared_storage_and_excess_residency():
    owner = model()
    owner.text_model.layers[2].weight = owner.text_model.layers[0].weight
    with pytest.raises(ValueError, match="storage alias"):
        EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(resident_layers=2))
    with pytest.raises(ValueError, match="resident_layers"):
        EncoderResidency(model(), torch.device("cpu"), EncoderResidencyOptions(resident_layers=100))
