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

    def __post_init__(self) -> None:
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
        ):
            value = env.get("VLLM_OMNI_ENCODER_" + name, "0")
            if value not in ("0", "1"):
                raise ValueError("VLLM_OMNI_ENCODER_" + name + " must be 0 or 1")
            values[field] = value == "1"
        return cls(**values, resident_layers=int(env.get("VLLM_OMNI_ENCODER_RESIDENT_LAYERS", "0")))

    @property
    def enabled(self) -> bool:
        return (
            self.frozen_master
            or self.cache_retention
            or self.pinned_buffers
            or self.resident_layers != 0
            or self.packed_transport
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
        self._validate_partition_aliases(partitions)
        use_masters = (
            options.frozen_master or options.pinned_buffers or options.resident_layers != 0 or options.packed_transport
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
        try:
            for stager in self.stagers:
                stager.load()
        except BaseException:
            self.offload()
            raise
        self.stats["loads"] += 1

    def offload(self) -> None:
        if not self.stagers:
            self.module.to("cpu")
            if self.cache is not None:
                self.cache.release_if_needed()
            else:
                current_omni_platform.empty_cache()
        for stager in self.stagers:
            stager.offload()
        self.stats["offloads"] += 1
