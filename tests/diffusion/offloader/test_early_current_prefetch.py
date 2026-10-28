# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadHook

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


def make_protocol():
    hook = DistributedLayerwiseOffloadHook.__new__(DistributedLayerwiseOffloadHook)
    calls = []
    hook.dp_size = 1
    hook.current_slot = 0
    hook._is_group_first = True
    hook._group_id = 7
    hook._shared_slot_group = [-1, -1]
    hook._shared_slot_owner = [-1, -1]
    hook._materialization_probe = None
    prev = SimpleNamespace(_prefetched_slot=1)

    def copy(slot, non_blocking):
        calls.append(("copy", slot, non_blocking))
        hook._shared_slot_group[slot] = 7
        hook._shared_slot_owner[slot] = id(prev)

    prev.prefetch_layer = copy
    prev._wait_for_weights = lambda slot: calls.append(("wait", slot))
    hook._prev_hook = prev
    hook.prefetch_layer = lambda slot, non_blocking: calls.append(("next", slot, non_blocking))
    return hook, calls


def test_early_copy_uses_async_then_forward_waits_without_recopied_weights():
    hook, calls = make_protocol()
    assert hook.prefetch_current_layer()
    assert calls == [("copy", 1, True)]
    hook.pre_forward(nn.Identity())
    assert calls == [("copy", 1, True), ("wait", 1), ("next", 0, True)]


def test_overwrite_after_early_prefetch_falls_back_to_original_sync_restore():
    hook, calls = make_protocol()
    hook.prefetch_current_layer()
    hook._shared_slot_owner[1] = -1
    hook.pre_forward(nn.Identity())
    assert calls[1] == ("copy", 1, False)
    assert calls[2] == ("wait", 1)


def test_early_copy_skips_allgather_and_missing_slot_protocol():
    hook, calls = make_protocol()
    hook.dp_size = 8
    assert not hook.prefetch_current_layer()
    hook.dp_size = 1
    hook._shared_slot_group = None
    assert not hook.prefetch_current_layer()
    assert calls == []
