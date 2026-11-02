# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in experiments for frozen MiniMax-H3 encoder storage ownership.

Options are read once before the first transition. All defaults preserve legacy
placement. Masters include buffers: these experiments require immutable inference
weights and buffers and must not be used for training or stateful encoders.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

from .module_residency import BoundedAllocatorCache, PinnedModuleStager


@dataclass(frozen=True)
class EncoderResidencyOptions:
    frozen_master: bool = False
    cache_retention: bool = False
    pinned_buffers: bool = False
    packed_transport: bool = False
    resident_layers: int = 0
    group_size: int = 0
    prefetch: bool = False

    def __post_init__(self) -> None:
        if self.prefetch and (not self.pinned_buffers or self.group_size < 1):
            raise ValueError("prefetch requires PINNED_BUFFERS=1 and GROUP_SIZE>=1")
        if self.group_size < 0:
            raise ValueError("group_size must be 0 (model), 1 (layer), or a positive block count")
        if self.group_size and self.resident_layers == -1:
            raise ValueError("Model residency and grouped unloading cannot be combined")
        if self.resident_layers < -1:
            raise ValueError("resident_layers must be -1 (all), 0 (off), or a positive layer count")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> EncoderResidencyOptions:
        env = os.environ if environ is None else environ
        values = {}
        for field, name in (
            ("frozen_master", "FROZEN_MASTER"),
            ("cache_retention", "CACHE_RETENTION"),
            ("pinned_buffers", "PINNED_BUFFERS"),
            ("packed_transport", "PACKED_TRANSPORT"),
            ("prefetch", "PREFETCH"),
        ):
            value = env.get("VLLM_OMNI_ENCODER_" + name, "0")
            if value not in ("0", "1"):
                raise ValueError("VLLM_OMNI_ENCODER_" + name + " must be 0 or 1")
            values[field] = value == "1"
        return cls(
            **values,
            resident_layers=int(env.get("VLLM_OMNI_ENCODER_RESIDENT_LAYERS", "0")),
            group_size=int(env.get("VLLM_OMNI_ENCODER_GROUP_SIZE", "0")),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.frozen_master
            or self.cache_retention
            or self.pinned_buffers
            or self.resident_layers != 0
            or self.packed_transport
            or self.group_size != 0
        )


class EncoderResidency:
    """Own the immutable CPU masters for a full encoder, without D2H offload."""

    def __init__(self, module: nn.Module, device: torch.device, options: EncoderResidencyOptions) -> None:
        self.options = options
        self.module = module
        self.device = device
        self.cache = BoundedAllocatorCache(device) if options.cache_retention else None
        self.stats = {"loads": 0, "offloads": 0}
        partitions: list[tuple[list[nn.Module], bool]] = [([module], options.resident_layers == -1)]
        if options.resident_layers > 0:
            layers = list(module.text_model.layers)
            if options.resident_layers > len(layers):
                raise ValueError("resident_layers exceeds the encoder layer count")
            count = options.resident_layers
            other = [module.vision, *[child for name, child in module.text_model.named_children() if name != "layers"]]
            partitions = [(layers[:count], True), ([*other, *layers[count:]], False)]
        self.block_groups: list[tuple[list[nn.Module], int]] = []
        if options.group_size:
            text_layers = list(module.text_model.layers)
            count = options.resident_layers
            vision_layers = list(getattr(module.vision, "blocks", ()))
            nonblocks = [child for name, child in module.text_model.named_children() if name != "layers"]
            if vision_layers:
                nonblocks.extend(child for name, child in module.vision.named_children() if name != "blocks")
            else:
                nonblocks.append(module.vision)
            partitions = [(nonblocks, False)]
            if count:
                partitions.append((text_layers[:count], True))
            for stack in (vision_layers, text_layers[count:]):
                for start in range(0, len(stack), options.group_size):
                    members = stack[start : start + options.group_size]
                    self.block_groups.append((members, len(partitions)))
                    partitions.append((members, False))
        self._validate_partition_aliases(partitions)
        use_masters = (
            options.frozen_master
            or options.pinned_buffers
            or options.resident_layers != 0
            or options.packed_transport
            or options.group_size != 0
        )
        self.stagers = (
            [
                PinnedModuleStager(
                    members,
                    device,
                    pin_memory=options.pinned_buffers,
                    cache_retention=self.cache,
                    retain_device_storage=resident,
                    copy_back=not options.frozen_master,
                    coalesce=options.packed_transport,
                )
                for members, resident in partitions
            ]
            if use_masters
            else []
        )
        self._active = False
        self._handles = []
        for members, index in self.block_groups:
            self._handles.append(members[0].register_forward_pre_hook(self._before_group(index)))
            self._handles.append(members[-1].register_forward_hook(self._after_group(index), always_call=True))

    def _before_group(self, index: int):
        def before(_module, _args):
            if not self._active:
                raise RuntimeError("Call encoder.load_to_device before grouped encoder execution")
            self.stagers[index].load()
            if self.options.prefetch:
                indices = [group_index for _, group_index in self.block_groups]
                position = indices.index(index)
                if position + 1 < len(indices):
                    self.stagers[indices[position + 1]].prefetch()

        return before

    def _after_group(self, index: int):
        def after(_module, _args, _output):
            self.stagers[index].offload()

        return after

    def close(self) -> None:
        try:
            self.offload()
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    @staticmethod
    def _validate_partition_aliases(partitions: list[tuple[list[nn.Module], bool]]) -> None:
        owners: dict[tuple, int] = {}
        for index, (members, _) in enumerate(partitions):
            for member in members:
                for target in (*member.parameters(), *member.buffers()):
                    tensor = PinnedModuleStager._local_tensor(target)
                    if tensor.is_meta:
                        raise ValueError("Encoder residency cannot snapshot meta tensors")
                    storage = tensor.untyped_storage()
                    if not storage.nbytes():
                        continue
                    key = (tensor.device, storage.data_ptr(), storage.nbytes())
                    if key in owners and owners[key] != index:
                        raise ValueError(
                            "Encoder storage alias crosses staging groups; use model grouping/full residency"
                        )
                    owners[key] = index

    def load(self) -> None:
        if not self.stagers:
            self.module.to(self.device)
        self._active = True
        block_indices = {index for _, index in self.block_groups}
        try:
            for index, stager in enumerate(self.stagers):
                if index not in block_indices:
                    stager.load()
        except BaseException:
            self.offload()
            raise
        self.stats["loads"] += 1

    def offload(self) -> None:
        self._active = False
        if not self.stagers:
            self.module.to("cpu")
            if self.cache is not None:
                self.cache.release_if_needed()
            else:
                current_omni_platform.empty_cache()
        for stager in self.stagers:
            stager.offload()
        self.stats["offloads"] += 1
