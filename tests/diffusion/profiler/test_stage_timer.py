# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable tests for non-synchronizing stage accounting, without the serving stack."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/profiler/stage_timer.py"
    spec = importlib.util.spec_from_file_location("stage_timer_test_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 0.0
        self.pending = 0.0
        self.sync_count = 0

    def synchronize(self):
        self.now += self.pending
        self.pending = 0.0
        self.sync_count += 1

    def __call__(self):
        return self.now


def test_stage_does_not_wait_for_pending_device_work(api, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(api.time, "perf_counter", clock)
    timer = api.StageTimer()
    clock.pending = 20.0
    with timer.activate():
        with api.timed_stage("encoder_forward"):
            clock.now += 1.0
            clock.pending += 2.0
        with api.timed_stage("result_to_cpu"):
            clock.now += 4.0
    assert timer.durations == {"encoder_forward": 1.0, "result_to_cpu": 4.0}
    assert clock.pending == 22.0
    assert clock.sync_count == 0


def test_inactive_stage_does_not_read_clock_or_sync(api, monkeypatch):
    def unexpected():
        pytest.fail("disabled timing must not touch the clock")

    monkeypatch.setattr(api.time, "perf_counter", unexpected)
    with api.timed_stage("encoder_forward"):
        value = 42
    assert value == 42


def test_exception_resets_scope_and_keeps_partial_timing(api, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(api.time, "perf_counter", clock)
    timer = api.StageTimer()
    with pytest.raises(ValueError, match="forward failed"):
        with timer.activate():
            with api.timed_stage("encoder_forward"):
                clock.now += 2.0
                raise ValueError("forward failed")
    assert timer.durations == {"encoder_forward": 2.0}
    sync_count = clock.sync_count
    with api.timed_stage("encoder_forward"):
        clock.now += 5.0
    assert clock.sync_count == sync_count
    assert timer.durations == {"encoder_forward": 2.0}


def test_component_migrations_exclude_unrelated_modules(api, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(api.time, "perf_counter", clock)
    encoder, dit = object(), object()
    timer = api.StageTimer(
        component=encoder,
        component_stages={"load": "encoder_load", "offload": "encoder_offload", "release": "device_memory_release"},
    )
    with timer.activate():
        with api.timed_stage("offload", component=dit):
            clock.now += 100.0
        with api.timed_stage("load", component=encoder):
            clock.now += 1.0
        with api.timed_stage("offload", component=encoder):
            clock.now += 2.0
        with api.timed_stage("release", component=encoder):
            clock.now += 3.0
    assert timer.durations == {"encoder_load": 1.0, "encoder_offload": 2.0, "device_memory_release": 3.0}
    assert clock.sync_count == 0


def test_nested_stages_are_exclusive_and_repeated_stages_accumulate(api, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(api.time, "perf_counter", clock)
    timer = api.StageTimer()
    with timer.activate():
        with api.timed_stage("encoder_offload"):
            clock.now += 2.0
            with api.timed_stage("device_memory_release"):
                clock.now += 3.0
            clock.now += 4.0
        with api.timed_stage("encoder_offload"):
            clock.now += 1.0
    assert timer.durations == {"encoder_offload": 7.0, "device_memory_release": 3.0}


def test_nested_scopes_and_new_requests_do_not_leak_records(api, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(api.time, "perf_counter", clock)
    outer, inner, next_request = (api.StageTimer() for _ in range(3))
    with outer.activate():
        with inner.activate():
            with api.timed_stage("encoder_forward"):
                clock.now += 2.0
        with api.timed_stage("encoder_forward"):
            clock.now += 3.0
    with next_request.activate():
        with api.timed_stage("encoder_forward"):
            clock.now += 4.0
    assert outer.durations == {"encoder_forward": 3.0}
    assert inner.durations == {"encoder_forward": 2.0}
    assert next_request.durations == {"encoder_forward": 4.0}
