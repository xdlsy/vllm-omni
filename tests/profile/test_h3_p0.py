# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only checks for optional profiling without importing the serving stack."""

import ast
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from contextlib import nullcontext, suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_profiler():
    path = Path(__file__).parents[2] / "vllm_omni/profiler/h3_p0.py"
    spec = importlib.util.spec_from_file_location("h3_p0_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class H3P0Tests(unittest.TestCase):
    def setUp(self):
        self.p = load_profiler()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"VLLM_OMNI_H3_P0_PROFILE_DIR": self.tmp.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def records(self):
        return [json.loads(line) for p in Path(self.tmp.name).glob("*.jsonl") for line in p.read_text().splitlines()]

    def test_disabled_does_not_touch_metadata_or_output(self):
        os.environ.pop("VLLM_OMNI_H3_P0_PROFILE_DIR")
        with self.p.request("r"):
            self.assertEqual(self.p.call("work", lambda: 42), 42)
        self.assertEqual(self.records(), [])

    def test_nested_spans_keep_identity_parent_and_exception(self):
        with self.p.request("r", global_rank=3):
            with self.p.stage("outer"):
                with self.assertRaisesRegex(ValueError, "original"):
                    self.p.call("inner", lambda: (_ for _ in ()).throw(ValueError("original")))
        records = self.records()
        outer = next(r for r in records if r["stage"] == "outer")
        inner = next(r for r in records if r["stage"] == "inner")
        self.assertEqual(inner["parent_span_id"], outer["span_id"])
        self.assertEqual(inner["status"], "error")
        self.assertEqual(inner["request_id"], "r")
        self.assertEqual(inner["global_rank"], 3)
        self.assertLessEqual(inner["begin_ns"], inner["end_ns"])

    def test_tensor_metadata_never_reads_device_values(self):
        class Tensor:
            shape = (2, 3)
            dtype = "float32"
            device = "npu:2"

            def stride(self):
                return (3, 1)

            def is_contiguous(self):
                return True

            def numel(self):
                return 6

            def element_size(self):
                return 4

            def cpu(self):
                raise AssertionError("device read")

            item = tolist = cpu

        with self.p.request("r"):
            self.p.call("tensor", lambda: Tensor())
        meta = next(r for r in self.records() if r["stage"] == "tensor")["output"]
        self.assertEqual(meta["shape"], [2, 3])
        self.assertEqual(meta["logical_bytes"], 24)

    def test_async_requests_do_not_share_parent_or_request(self):
        async def task(rid):
            with self.p.request(rid):
                with self.p.stage("async"):
                    await asyncio.sleep(0)

        async def run():
            await asyncio.gather(task("a"), task("b"))

        asyncio.run(run())
        spans = [r for r in self.records() if r["stage"] == "async"]
        self.assertEqual({r["request_id"] for r in spans}, {"a", "b"})
        self.assertEqual(len({r["parent_span_id"] for r in spans}), 2)

    def test_cancelled_request_closes_span_and_preserves_cancellation(self):
        async def run():
            ready = asyncio.Event()

            async def task():
                with self.p.request("cancelled"):
                    with self.p.stage("wait", trace=False):
                        ready.set()
                        await asyncio.Future()

            pending = asyncio.create_task(task())
            await ready.wait()
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending

        asyncio.run(run())
        record = next(r for r in self.records() if r["stage"] == "wait")
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_type"], "CancelledError")

    def test_first_dit_stops_new_spans_but_closes_open_spans(self):
        with self.p.request("r"):
            with self.p.stage("prepare"):
                self.p.first_dit(step=0)
                self.p.call("later_step", lambda: None)
        stages = [r["stage"] for r in self.records()]
        self.assertIn("dit/first_model_enter", stages)
        self.assertIn("prepare", stages)
        self.assertNotIn("later_step", stages)

    def test_buffer_limit_reports_drops_and_is_bounded(self):
        with self.p.request("r", max_events=3):
            for _ in range(20):
                self.p.call("work", lambda: None)
        records = self.records()
        summary = next(r for r in records if r["stage"] == "profile/summary")
        self.assertGreater(summary["dropped_events"], 0)
        self.assertLessEqual(len(records), 4)

    def test_write_failure_does_not_mask_model_error(self):
        with (
            self.assertLogs(self.p._logger, level="WARNING"),
            patch.object(Path, "mkdir", side_effect=OSError("disk unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                with self.p.request("r"):
                    raise RuntimeError("model failure")

    def test_context_wrapper_preserves_suppression_and_media_identity(self):
        with self.p.request("r"):
            with self.p.stage("image", media_id=4):
                with self.p.scope("rng", suppress(ValueError)):
                    raise ValueError("suppressed")
        exits = [r for r in self.records() if r["stage"] == "rng/exit"]
        self.assertEqual(exits[0]["media_id"], 4)
        self.assertEqual(exits[0]["status"], "ok")

    def test_detached_task_gets_fresh_buffer_linked_to_api_trace(self):
        async def run():
            ready = asyncio.Event()

            async def job():
                await ready.wait()
                with self.p.request("worker", force=True):
                    self.p.call("job", lambda: 1)

            with self.p.request("api"):
                task = asyncio.create_task(job())
            ready.set()
            await task

        asyncio.run(run())
        api = next(r for r in self.records() if r["request_id"] == "api")
        job = next(r for r in self.records() if r["stage"] == "job")
        self.assertEqual(job["parent_trace_id"], api["trace_id"])
        self.assertNotEqual(job["trace_id"], api["trace_id"])

    def test_torch_annotations_do_not_synchronize(self):
        annotations = []

        def record(name):
            annotations.append(name)
            return nullcontext()

        torch = SimpleNamespace(
            profiler=SimpleNamespace(record_function=record),
            accelerator=SimpleNamespace(synchronize=lambda: self.fail("unexpected device sync")),
        )
        with patch.dict(sys.modules, {"torch": torch}), self.p.request("r"):
            self.p.call("work", lambda: 7)
        self.assertEqual(annotations, ["H3_STAGE/work"])

    def test_async_upload_logs_length_without_payload(self):
        @self.p.profile("api/ref_read")
        async def upload():
            return b"private-pixel-payload"

        async def run():
            with self.p.request("r"):
                self.assertEqual(await upload(), b"private-pixel-payload")

        asyncio.run(run())
        record = next(r for r in self.records() if r["stage"] == "api/ref_read")
        self.assertEqual(record["output"]["length"], len(b"private-pixel-payload"))
        self.assertNotIn("private-pixel-payload", json.dumps(self.records()))

    def test_unsupported_tensor_layout_does_not_break_inference(self):
        class Tensor:
            shape, dtype, device = (3,), "float32", "npu:0"

            def stride(self):
                raise RuntimeError("unsupported layout")

        result = Tensor()
        with self.p.request("r"):
            self.assertIs(self.p.call("work", lambda: result), result)
        record = next(r for r in self.records() if r["stage"] == "work")
        self.assertTrue(record["output"]["metadata_unavailable"])

    def test_encode_prompt_keeps_processor_order_and_return_identity(self):
        path = Path(__file__).parents[2] / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "encode_prompt")
        ids, tags, hidden = object(), object(), object()
        calls = []

        class Processor:
            merge_size = 2

            def __call__(self, **kwargs):
                calls.append("processor")
                return {"pixel_values": object(), "image_grid_thw": object()}

        def distribute(value, vision):
            self.assertIs(value, ids)
            self.assertIn("pixel_values", vision)
            calls.append("distribute")
            return value

        def encode(value, vision):
            self.assertIs(value, ids)
            calls.append("encode")
            return hidden

        namespace = {
            "_h3_profile": self.p.profile,
            "_h3_call": self.p.call,
            "_dit_rank_world": lambda: (None, 0, 1),
            "_broadcast_rank0_exception": lambda error: self.assertIsNone(error),
            "build_minimax_h3_presentation": lambda *args, **kwargs: (ids, tags),
            "_broadcast_tensor": lambda value, **kwargs: value,
            "torch": SimpleNamespace(bfloat16="bf16", long="long"),
        }
        exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
        model = SimpleNamespace(
            processor=SimpleNamespace(image_processor=Processor()),
            text_encoder_tp_size=1,
            device="cpu",
            tokenizer=object(),
            _distribute_encode_inputs=distribute,
            _encode_text_hidden=encode,
        )
        prepared = SimpleNamespace(
            images=[object()],
            qwen_videos=[],
            prompt="private prompt",
            media=SimpleNamespace(task="ref2va"),
            condition_labels=[],
            video_timestamps=[],
        )
        with self.p.request("r"):
            result = namespace["encode_prompt"](model, prepared)
        self.assertIs(result[0], hidden)
        self.assertIs(result[1], tags)
        self.assertEqual(calls, ["processor", "distribute", "encode"])
        self.assertNotIn("private prompt", json.dumps(self.records()))

    def test_async_omni_exports_external_to_worker_request_id_mapping(self):
        path = Path(__file__).parents[2] / "vllm_omni/entrypoints/async_omni.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate")
        namespace = {"_h3_event": self.p.event, "AsyncGenerator": AsyncGenerator}
        exec("from __future__ import annotations\n" + ast.unparse(function), namespace)

        class StopAfterMapping:
            async def __aenter__(self):
                raise RuntimeError("stop after mapping")

            async def __aexit__(self, *exc):
                return False

        model = SimpleNamespace(_get_unique_request_id=lambda rid: rid + "-internal", _pause_cond=StopAfterMapping())

        async def run():
            with self.p.request("external"):
                with self.assertRaisesRegex(RuntimeError, "stop after mapping"):
                    async for _ in namespace["generate"](model, "prompt", request_id="external"):
                        pass

        asyncio.run(run())
        mappings = [r for r in self.records() if r["stage"] == "engine/request_id_map"]
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0]["external_request_id"], "external")
        self.assertEqual(mappings[0]["internal_request_id"], "external-internal")


if __name__ == "__main__":
    unittest.main()
