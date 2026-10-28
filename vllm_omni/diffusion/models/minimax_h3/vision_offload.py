# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional vision entry points into the existing DLO ownership protocol."""

from torch import nn


def prefetch_first_block(block: nn.Module) -> None:
    registry = getattr(block, "_hook_registry", None)
    if registry is None:
        return
    hook = registry.get_hook("distributed_layerwise_offload")
    if hook is not None:
        hook.prefetch_current_layer()
