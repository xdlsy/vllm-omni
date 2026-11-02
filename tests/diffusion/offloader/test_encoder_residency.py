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
    first_copies = retained[0].stats["h2d_copies"]
    residency.load()
    assert ptrs == [storage.data_ptr() for storage in retained[0]._device_storages]
    assert retained[0].stats["h2d_copies"] == first_copies
    assert retained[0].stats["d2h_copies"] == 0


def test_partition_rejects_shared_storage_and_excess_residency():
    owner = model()
    owner.text_model.layers[2].weight = owner.text_model.layers[0].weight
    with pytest.raises(ValueError, match="storage alias"):
        EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(resident_layers=2))
    with pytest.raises(ValueError, match="resident_layers"):
        EncoderResidency(model(), torch.device("cpu"), EncoderResidencyOptions(resident_layers=100))


def test_pinning_without_frozen_master_copies_back_device_updates():
    with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor.clone()):
        owner = model()
        residency = EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(pinned_buffers=True))
    residency.load()
    owner.text_model.layers[0].weight.data.fill_(42)
    residency.offload()
    assert torch.all(owner.text_model.layers[0].weight == 42)


def test_packed_transport_preserves_mixed_dtype_strided_aliases():
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager

    owner = nn.Module()
    base = torch.arange(24, dtype=torch.float32)
    owner.left = nn.Parameter(base.as_strided((3, 3), (4, 1), 1))
    owner.right = nn.Parameter(base.as_strided((3, 2), (4, 1), 3))
    owner.register_buffer("integers", torch.arange(7, dtype=torch.int64))
    owner.register_buffer("empty", torch.empty(0))
    expected = {name: tensor.detach().clone() for name, tensor in (*owner.named_parameters(), *owner.named_buffers())}
    layout = (owner.left.stride(), owner.left.storage_offset(), owner.right.storage_offset())
    stager = PinnedModuleStager(owner, torch.device("cpu"), pin_memory=False, coalesce=True)
    for _ in range(2):
        stager.load()
        assert owner.left.untyped_storage().data_ptr() == owner.right.untyped_storage().data_ptr()
        assert layout == (owner.left.stride(), owner.left.storage_offset(), owner.right.storage_offset())
        for name, tensor in (*owner.named_parameters(), *owner.named_buffers()):
            assert torch.equal(tensor, expected[name])
        stager.offload()
    assert stager.stats["h2d_copies"] == 2
    assert stager.stats["h2d_bytes"] == 2 * stager._transport_masters[0].numel()


@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_grouped_layer_hooks_match_reference_and_unload(group_size):
    owner = model()
    value = torch.randn(2, 3)
    expected = value
    for layer in owner.text_model.layers:
        expected = layer(expected)
    residency = EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(group_size=group_size))
    residency.load()
    output = value
    for index, layer in enumerate(owner.text_model.layers):
        output = layer(output)
        if (index + 1) % group_size == 0:
            assert not residency.stagers[residency.block_groups[index // group_size][1]].loaded
    assert torch.equal(output, expected)
    residency.offload()
    assert not any(stager.loaded for stager in residency.stagers)
    residency.close()
    assert not any(layer._forward_pre_hooks or layer._forward_hooks for layer in owner.text_model.layers)


def test_group_exception_cleanup_allows_next_request():
    owner = model()
    residency = EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(group_size=2))
    residency.load()
    owner.text_model.layers[0](torch.ones(1, 3))
    residency.offload()  # exception before group's last layer
    assert not any(stager.loaded for stager in residency.stagers)
    residency.load()
    for layer in owner.text_model.layers:
        layer(torch.ones(1, 3))
    residency.offload()


def test_prefetch_does_not_rebind_or_wait_until_consumed():
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager, current_omni_platform

    owner = nn.Linear(3, 3)
    stager = PinnedModuleStager(owner, torch.device("cpu"), pin_memory=False)
    master = owner.weight.data_ptr()
    with patch.object(current_omni_platform, "current_stream") as stream:
        stager.prefetch()
        stager.prefetch()
        assert owner.weight.data_ptr() == master
        assert stager._pending_storages
        stream.return_value.wait_event.assert_not_called()
        stager.load()
        assert owner.weight.data_ptr() != master
        assert not stager._pending_storages
        stream.return_value.wait_event.assert_called_once()
    stager.offload()


def test_prefetch_unconsumed_storage_is_drained_on_failure_cleanup():
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager

    stager = PinnedModuleStager(nn.Linear(3, 3), torch.device("cpu"), pin_memory=False)
    stager.prefetch()
    stager.offload()
    assert not stager._pending_storages
    assert not stager.loaded


def test_prefetch_requires_grouping_and_pinned_buffers():
    with pytest.raises(ValueError, match="prefetch"):
        EncoderResidencyOptions(prefetch=True)
    with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor.clone()):
        owner = model()
        residency = EncoderResidency(
            owner, torch.device("cpu"), EncoderResidencyOptions(prefetch=True, pinned_buffers=True, group_size=1)
        )
    residency.load()
    owner.text_model.layers[0](torch.ones(1, 3))
    next_index = residency.block_groups[1][1]
    assert residency.stagers[next_index]._pending_storages
    residency.offload()
    assert not any(stager._pending_storages for stager in residency.stagers)


def test_reusable_buffers_keep_addresses_but_repeat_h2d_and_refresh_masters():
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager

    owner = nn.Linear(3, 3)
    stager = PinnedModuleStager(owner, torch.device("cpu"), pin_memory=False, reuse_device_buffers=True)
    stager.load()
    pointer = owner.weight.data_ptr()
    stager.offload()
    owner.weight.data.fill_(17)
    stager.load()
    assert owner.weight.data_ptr() == pointer
    assert torch.all(owner.weight == 17)
    assert stager.stats["buffer_reuses"] == 1
    assert stager.stats["h2d_copies"] == 2 * len(stager._groups)
    assert stager.stats["device_allocations"] == len(stager._groups)
    stager.close()
    assert not stager._reusable_transports
    assert not stager._device_storages


def test_resident_and_reusable_storage_are_not_conflated():
    with pytest.raises(ValueError, match="reuse"):
        EncoderResidencyOptions(resident_layers=1, reuse_buffers=True)
    options = EncoderResidencyOptions.from_env({"VLLM_OMNI_ENCODER_REUSE_BUFFERS": "1"})
    residency = EncoderResidency(model(), torch.device("cpu"), options)
    assert residency.stagers
    residency.load()
    residency.offload()
    assert residency.transport_stats["device_allocations"] > 0
    assert residency.transport_stats["d2h_copies"] > 0
    residency.close()
    assert not any(stager._reusable_transports for stager in residency.stagers)


def test_controller_attempts_all_group_cleanup_after_one_error():
    owner = model()
    residency = EncoderResidency(owner, torch.device("cpu"), EncoderResidencyOptions(group_size=1))
    residency.load()
    with (
        patch.object(residency.stagers[0], "offload", side_effect=RuntimeError("cleanup failed")),
        patch.object(residency.stagers[-1], "offload", wraps=residency.stagers[-1].offload) as last,
    ):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            residency.offload()
        last.assert_called_once()


@pytest.mark.parametrize("copy_back", [False, True])
@pytest.mark.parametrize("coalesce", [False, True])
def test_reuse_actual_allocation_count_across_three_requests(copy_back, coalesce):
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager

    owner = nn.Linear(3, 3)
    stager = PinnedModuleStager(
        owner, torch.device("cpu"), pin_memory=False, reuse_device_buffers=True, coalesce=coalesce, copy_back=copy_back
    )
    count = len(stager._transport_masters)
    with patch.object(torch, "empty_like", wraps=torch.empty_like) as allocate:
        pointers = []
        for _ in range(3):
            stager.load()
            pointers.append(owner.weight.data_ptr())
            stager.offload()
        assert len(set(pointers)) == 1
        assert allocate.call_count == count
    assert stager.stats["device_allocations"] == count
    assert stager.stats["h2d_copies"] == 3 * count
    assert stager.stats["d2h_copies"] == (3 * count if copy_back else 0)
    stager.close()


def test_failed_synchronization_keeps_storage_owned_for_retry():
    from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager, current_omni_platform

    stager = PinnedModuleStager(nn.Linear(3, 3), torch.device("cpu"), pin_memory=False)
    stager.load()
    storages = stager._device_storages
    with patch.object(current_omni_platform, "synchronize", side_effect=RuntimeError("device lost")):
        with pytest.raises(RuntimeError, match="device lost"):
            stager.offload()
    assert stager.loaded
    assert stager._device_storages is storages
    assert storages
    stager.close()
