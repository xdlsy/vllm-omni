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

from .module_residency import PinnedModuleStager


@dataclass(frozen=True)
class EncoderResidencyOptions:
    frozen_master: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> EncoderResidencyOptions:
        env = os.environ if environ is None else environ
        value = env.get("VLLM_OMNI_ENCODER_FROZEN_MASTER", "0")
        if value not in ("0", "1"):
            raise ValueError("VLLM_OMNI_ENCODER_FROZEN_MASTER must be 0 or 1")
        return cls(frozen_master=value == "1")

    @property
    def enabled(self) -> bool:
        return self.frozen_master


class EncoderResidency:
    """Own the immutable CPU masters for a full encoder, without D2H offload."""

    def __init__(self, module: nn.Module, device: torch.device, options: EncoderResidencyOptions) -> None:
        self.options = options
        self.stats = {"loads": 0, "offloads": 0}
        self.stagers = [PinnedModuleStager(module, device, pin_memory=False)]

    def load(self) -> None:
        try:
            for stager in self.stagers:
                stager.load()
        except BaseException:
            self.offload()
            raise
        self.stats["loads"] += 1

    def offload(self) -> None:
        for stager in self.stagers:
            stager.offload()
        self.stats["offloads"] += 1
