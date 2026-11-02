# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage accounting preserves the encoder's output and offload lifecycle."""

import json
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline(monkeypatch):
    value = object.__new__(module.MiniMaxH3Pipeline)
    nn.Module.__init__(value)
    value.device = torch.device("cpu")
    value.enable_diffusion_pipeline_profiler = True
    value._profiler_lock = Lock()
    value._stage_durations = {}
    value.text_encoder_tp_size = 1
    encoder = MiniMaxH3Qwen3VLEncoder("unused", device=value.device, load_model=False)
    encoder.vision = nn.Linear(1, 1)
    encoder.text_model = nn.Linear(1, 1)
    value.text_encoder = encoder
    value.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    value.tokenizer = object()
    value._model_cpu_offload_modules = []
    monkeypatch.setattr(value, "_uses_manual_component_offload", lambda component: True)
    monkeypatch.setattr(value, "_distribute_encode_inputs", lambda ids, vision: ids)
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_rank0_exception", lambda exc: (_ for _ in ()).throw(exc) if exc else None)
    monkeypatch.setattr(module, "_broadcast_tensor", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(
        module, "build_minimax_h3_presentation", lambda *args, **kwargs: (torch.tensor([1, 2]), torch.tensor([0, 0]))
    )
    monkeypatch.setattr(encoder, "_encode", lambda *args, **kwargs: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: False)
    return value


def _prepared():
    return SimpleNamespace(
        images=(),
        qwen_videos=(),
        prompt="test",
        media=SimpleNamespace(task="t2va"),
        condition_labels=(),
        video_timestamps=(),
    )


def _stages(pipeline):
    prefix = "MiniMaxH3Pipeline.encode_prompt."
    return {name.removeprefix(prefix) for name in pipeline.stage_durations}


def test_manual_encode_records_seven_stages_and_preserves_output(pipeline):
    hidden, tags = pipeline.encode_prompt(_prepared())
    assert torch.equal(hidden, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16))
    assert torch.equal(tags, torch.tensor([0, 0]))
    assert _stages(pipeline) == {
        "input_prepare_and_distribute",
        "encoder_load",
        "encoder_forward",
        "result_to_cpu",
        "encoder_offload",
        "device_memory_release",
        "result_distribute",
    }
    assert all(duration >= 0 for duration in pipeline.stage_durations.values())


def test_disabled_encode_does_not_add_sync_or_records(pipeline, monkeypatch):
    pipeline.enable_diffusion_pipeline_profiler = False
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: True)

    def unexpected():
        pytest.fail("disabled profiling must not synchronize")

    monkeypatch.setattr(module.current_omni_platform, "synchronize", unexpected)
    hidden, tags = pipeline.encode_prompt(_prepared())
    assert hidden.shape == (1, 2)
    assert tags.shape == (2,)
    assert pipeline._stage_durations == {}


def test_resident_encoder_has_no_offload_measurement(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "_uses_manual_component_offload", lambda component: False)
    pipeline.encode_prompt(_prepared())
    assert "encoder_load" in _stages(pipeline)
    assert "encoder_offload" not in _stages(pipeline)
    assert "device_memory_release" not in _stages(pipeline)


def test_non_encoder_rank_only_records_preparation_and_distribution(pipeline, monkeypatch):
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 1, 2))
    pipeline.encode_prompt(None)
    assert _stages(pipeline) == {"input_prepare_and_distribute", "result_distribute"}


def test_failure_reports_partial_stages_and_restores_timing_scope(pipeline, monkeypatch):
    reports = []
    monkeypatch.setattr(module.logger, "info", lambda fmt, payload: reports.append(json.loads(payload)))

    def fail(*args, **kwargs):
        raise ValueError("encoder failed")

    monkeypatch.setattr(pipeline.text_encoder, "_encode", fail)
    with pytest.raises(ValueError, match="encoder failed"):
        pipeline.encode_prompt(_prepared())
    assert reports[-1]["status"] == "failed"
    assert reports[-1]["rank"] == 0
    assert "encoder_forward" in _stages(pipeline)
    assert "encoder_offload" in _stages(pipeline)
    assert "result_distribute" not in _stages(pipeline)
    before = pipeline.stage_durations
    pipeline.text_encoder.load_to_device()
    assert pipeline.stage_durations == before


def test_sequential_hook_load_is_measured_without_forcing_encoder_offload(pipeline, monkeypatch):
    hook = SequentialOffloadHook([], pipeline.device, pin_memory=False)
    # Exercise the real migration path even though the fixture's weights are CPU.
    # Calling through a small module keeps the hook implementation real without
    # requiring a model checkpoint or an accelerator.
    encoder = pipeline.text_encoder
    encoder.register_forward_pre_hook(lambda component, args: hook.pre_forward(component, *args)[0])
    pipeline._model_cpu_offload_modules = [encoder]
    monkeypatch.setattr(module.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(module.current_omni_platform, "get_free_memory", lambda: 1024**3)
    pipeline.encode_prompt(_prepared())
    assert "encoder_load" in _stages(pipeline)
    assert "encoder_forward" in _stages(pipeline)
    assert "encoder_offload" not in _stages(pipeline)


def test_each_summary_is_per_call_while_request_records_accumulate(pipeline, monkeypatch):
    reports = []
    monkeypatch.setattr(module.logger, "info", lambda fmt, payload: reports.append(json.loads(payload)))
    pipeline.encode_prompt(_prepared())
    pipeline.encode_prompt(_prepared())
    assert len(reports) == 2
    for report in reports:
        assert report["rank"] == 0
        assert report["status"] == "ok"
        assert "encode_prompt" not in report
        assert report["timing_mode"] == "host_wall"
        assert set(report) == {"rank", "status", "timing_mode"} | _stages(pipeline)
    for stage in _stages(pipeline):
        assert pipeline.stage_durations[f"MiniMaxH3Pipeline.encode_prompt.{stage}"] == pytest.approx(
            reports[0][stage] + reports[1][stage]
        )


def test_enabled_encode_adds_no_device_sync(pipeline, monkeypatch):
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: True)

    def unexpected():
        pytest.fail("enabled timing must not add device synchronization")

    monkeypatch.setattr(module.current_omni_platform, "synchronize", unexpected)
    pipeline.encode_prompt(_prepared())
    assert "encoder_forward" in _stages(pipeline)


def test_outer_profiler_keeps_original_two_sync_calls(pipeline, monkeypatch):
    from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import profiler

    synchronizations = []
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(module.current_omni_platform, "synchronize", lambda: synchronizations.append(True))
    wrapped = profiler("MiniMaxH3Pipeline.encode_prompt", pipeline.encode_prompt, pipeline)
    hidden, _ = wrapped(_prepared())
    assert hidden.shape == (1, 2)
    assert len(synchronizations) == 2
    assert "MiniMaxH3Pipeline.encode_prompt" in pipeline.stage_durations


def test_other_pipeline_profiler_keeps_existing_sync_default(pipeline, monkeypatch):
    from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import profiler

    synchronizations = []
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(module.current_omni_platform, "synchronize", lambda: synchronizations.append(True))
    other = SimpleNamespace(_profiler_lock=Lock(), _stage_durations={})
    result = profiler("OtherPipeline.encode_prompt", lambda: 42, other)()
    assert result == 42
    assert len(synchronizations) == 2
    assert "OtherPipeline.encode_prompt" in other._stage_durations
