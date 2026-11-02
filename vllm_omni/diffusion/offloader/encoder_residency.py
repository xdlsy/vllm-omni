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

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> EncoderResidencyOptions:
        env = os.environ if environ is None else environ
        values = {}
        for field, name in (("frozen_master", "FROZEN_MASTER"), ("cache_retention", "CACHE_RETENTION")):
            value = env.get("VLLM_OMNI_ENCODER_" + name, "0")
            if value not in ("0", "1"):
                raise ValueError("VLLM_OMNI_ENCODER_" + name + " must be 0 or 1")
            values[field] = value == "1"
        return cls(**values)

    @property
    def enabled(self) -> bool:
        return self.frozen_master or self.cache_retention


class EncoderResidency:
    """Own the immutable CPU masters for a full encoder, without D2H offload."""

    def __init__(self, module: nn.Module, device: torch.device, options: EncoderResidencyOptions) -> None:
        self.options = options
        self.module = module
        self.device = device
        self.cache = BoundedAllocatorCache(device) if options.cache_retention else None
        self.stats = {"loads": 0, "offloads": 0}
        self.stagers = (
            [PinnedModuleStager(module, device, pin_memory=False, cache_retention=self.cache)]
            if options.frozen_master
            else []
        )

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
