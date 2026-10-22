# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the diagnostic contract; no serving/NPU imports required."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class VisionContractTests(unittest.TestCase):
    parallel: ModuleType
    probe: ModuleType

    @classmethod
    def setUpClass(cls):
        cls.parallel = load_file(
            "vision_parallel_probe_test", "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
        )
        cls.probe = load_file("vision_probe_test", "benchmarks/minimax_h3/vision_encoding_probe.py")

    def test_segment_merger_and_head_round_trip(self):
        # Uneven, empty and very short segments must retain original token order.
        for p in (1, 2, 4, 8, 16):
            plan = self.parallel.build_vision_plan((4, 0, 20, 64, 12, 8), p)
            ids = [i for rank in plan.rank_token_indices for i in rank]
            self.assertEqual(sorted(ids), list(range(108)))
            merged = [i for rank in plan.rank_merger_indices for i in rank]
            self.assertEqual(sorted(merged), list(range(27)))
            # Emulate token-to-head exchange and inverse with distinguishable values.
            x = torch.arange(108 * 16 * 2).reshape(108, 16, 2)
            shards = [x[list(idx)].clone() for idx in plan.rank_token_indices]
            heads = []
            for dest in range(p):
                full = torch.empty((108, 16 // p, 2), dtype=x.dtype)
                for idx, shard in zip(plan.rank_token_indices, shards):
                    full[list(idx)] = shard[:, dest * (16 // p) : (dest + 1) * (16 // p)]
                heads.append(full)
            restored = torch.cat(heads, dim=1)
            self.assertTrue(torch.equal(x, restored))
            for indices in plan.rank_token_indices:
                for start in range(0, len(indices), 4):
                    self.assertEqual(indices[start : start + 4], tuple(range(indices[start], indices[start] + 4)))

    def test_illegal_config_fails_closed(self):
        for lengths, sp, heads in [((5,), 2, 16), ((4,), 3, 16), ((-4,), 2, 16), ((4,), 16, 12)]:
            with self.assertRaises(ValueError):
                self.parallel.build_vision_plan(lengths, sp, num_heads=heads)
        with self.assertRaisesRegex(ValueError, "world"):
            self.parallel.validate_stage_layout(8, tuple(range(16)), tuple(range(8)), 16, 16)
        with self.assertRaisesRegex(ValueError, "DP"):
            self.parallel.validate_stage_layout(16, tuple(range(16)), tuple(range(8)), 16, 16, data_parallel_size=2)
        with self.assertRaisesRegex(ValueError, "downstream"):
            self.parallel.validate_stage_layout(16, tuple(range(16)), tuple(range(16)), 16, 16)

    def test_empty_plan(self):
        self.assertEqual(self.parallel.build_vision_plan((), 16).rank_token_indices, ((),) * 16)

    def test_short_sdpa_head_guard_preserves_head_order_and_long_path(self):
        with patch.dict(sys.modules, {"vision_encoding_probe": self.probe}):
            runner = load_file("vision_ulysses_guard_test", "benchmarks/minimax_h3/vision_ulysses_probe.py")
        for heads in (1, 2, 4, 8):
            q = torch.randn(1, heads, 7, 8)
            expected = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            actual = runner.segment_attention(q, q, q, short_min_heads=8)
            torch.testing.assert_close(actual, expected)
        q = torch.ones(1, 2, 1025, 8)
        with patch.object(runner.F, "scaled_dot_product_attention", return_value=q) as call:
            self.assertIs(runner.segment_attention(q, q, q, short_min_heads=8), q)
            self.assertIs(call.call_args.args[0], q)

    def test_padded_transport_roundtrip_excludes_padding(self):
        for size in (1, 2, 4, 8, 16):
            plan = self.parallel.build_vision_plan((4, 0, 20, 64, 12, 8), size)
            x = torch.arange(108 * 16 * 2).reshape(108, 16, 2)
            capacity = max(map(len, plan.rank_token_indices))
            outgoing = [self.parallel.pack_heads(x[list(ids)], size, capacity) for ids in plan.rank_token_indices]
            full_heads = [
                self.parallel.unpack_tokens(torch.stack([part[rank] for part in outgoing]), plan)
                for rank in range(size)
            ]
            for rank, full in enumerate(full_heads):
                self.assertTrue(torch.equal(full, x[:, rank * (16 // size) : (rank + 1) * (16 // size)]))
            inverse = [self.parallel.pack_tokens(full, plan) for full in full_heads]
            for rank, ids in enumerate(plan.rank_token_indices):
                received = torch.stack([part[rank] for part in inverse])
                local = self.parallel.unpack_heads(received, len(ids))
                self.assertTrue(torch.equal(local, x[list(ids)]))

    def test_actual_import_must_match_checkout(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.probe.verify_module_source(SimpleNamespace(__file__=__file__), "encoder.py")

    def test_local_shard_check_stops_on_inequality(self):
        with patch.dict(sys.modules, {"vision_encoding_probe": self.probe}):
            runner = load_file("vision_ulysses_probe_test", "benchmarks/minimax_h3/vision_ulysses_probe.py")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reference"
            full = torch.arange(24).reshape(8, 3)
            writer = self.probe.TensorCheckpoints(directory, reference=True)
            writer.record("point", full)
            writer.finish()
            check = runner.ShardChecks(directory, "cpu")
            with patch.object(runner.dist, "all_reduce"):
                check.check("point", full[[1, 3]], (1, 3))
                self.assertTrue(check.results["point"]["passed"])
                unequal = runner.ShardChecks(directory, "cpu")
                with self.assertRaisesRegex(RuntimeError, "point"):
                    unequal.check("point", full[[1, 3]] + 1, (1, 3))
                self.assertFalse(unequal.results["point"]["torch_equal"])
                self.assertEqual(unequal.results["point"]["first_mismatch"], [0, 0])
                self.assertNotIn("error", unequal.results["point"])

    def test_shard_gate_requires_all_points_and_rejects_duplicates(self):
        with patch.dict(sys.modules, {"vision_encoding_probe": self.probe}):
            runner = load_file("vision_ulysses_complete_test", "benchmarks/minimax_h3/vision_ulysses_probe.py")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reference"
            writer = self.probe.TensorCheckpoints(directory, reference=True)
            writer.record("a", torch.ones(2))
            writer.record("b", torch.ones(2))
            writer.finish()
            check = runner.ShardChecks(directory, "cpu")
            with patch.object(runner.dist, "all_reduce"):
                check.check("a", torch.ones(2), (0, 1))
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    check.finish()
                check.check("b", torch.ones(2), (0, 1))
                check.finish()
                with self.assertRaisesRegex(RuntimeError, "a"):
                    check.check("a", torch.ones(2), (0, 1))

    def test_case_must_be_covered_by_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            case = directory / "candidate.json"
            case.write_text(json.dumps({"configuration": {"task": "ref2va", "refs": []}}))
            (directory / "manifest.sha256.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.probe.validate_inputs(case, directory)

    def test_checkpoint_duplicate_extra_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reference"
            writer = self.probe.TensorCheckpoints(directory, reference=True)
            writer.record("final", torch.arange(4))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                writer.record("final", torch.arange(4))
            writer.finish()
            extra = self.probe.TensorCheckpoints(directory, reference=False)
            extra.record("unexpected", torch.arange(4))
            self.assertFalse(extra.finish()["passed"])
            torch.save(torch.zeros(4), directory / "00000.pt")
            corrupt = self.probe.TensorCheckpoints(directory, reference=False)
            with self.assertRaisesRegex(ValueError, "integrity"):
                corrupt.record("final", torch.arange(4))

    def test_scalar_empty_and_bfloat16(self):
        for tensor in (torch.tensor(1.0), torch.empty(0), torch.ones(3, dtype=torch.bfloat16)):
            self.assertTrue(self.probe.compare_tensor(tensor, tensor.clone())["passed"])

    def test_hooks_capture_blocks_and_deepstack_and_are_removed(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = torch.nn.Linear(2, 2)
                self.mlp = torch.nn.Linear(2, 2)

            def forward(self, x, *, cu_seqlens, position_embeddings):
                return self.mlp(self.attn(x))

        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = torch.nn.Identity()
                self.blocks = torch.nn.ModuleList([Block(), Block()])
                self.merger = torch.nn.Identity()
                self.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Identity()])

            def forward(self, x, grid):
                x = self.patch_embed(x)
                for block in self.blocks:
                    x = block(x, cu_seqlens=grid, position_embeddings=(grid, grid))
                return self.merger(x), [self.deepstack_merger_list[0](x)]

        model = Tower()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reference"
            pixels, grid = torch.ones(3, 2), torch.arange(3)
            self.probe.run_pass(model, pixels, grid, self.probe.TensorCheckpoints(directory, reference=True))
            check = self.probe.run_pass(model, pixels, grid, self.probe.TensorCheckpoints(directory, reference=False))
            self.assertTrue(check["passed"])
            for point in ("blocks.0", "blocks.1", "deepstack_merger_list.0", "final/1/0", "position_embeddings/0"):
                self.assertIn(point, check["points"])
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_tensor_report_rejects_tiny_difference_and_dtype(self):
        ref = torch.tensor([1.0, 2.0])
        changed = ref.clone()
        changed[1] = torch.nextafter(changed[1], torch.tensor(3.0))
        result = self.probe.compare_tensor(ref, changed)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_mismatch"], [1])
        self.assertFalse(self.probe.compare_tensor(ref, ref.double())["passed"])
        self.assertFalse(self.probe.compare_tensor(ref, ref.reshape(1, 2))["passed"])

    def test_bit_patterns_and_nonfinite_are_not_silently_equal(self):
        self.assertFalse(self.probe.compare_tensor(torch.tensor([0.0]), torch.tensor([-0.0]))["passed"])
        self.assertFalse(
            self.probe.compare_tensor(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"]
        )

    def test_checkpoint_manifest_requires_all_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "reference"
            writer = self.probe.TensorCheckpoints(directory, reference=True)
            writer.record("final", (torch.arange(4), [torch.ones(2)]))
            writer.finish()
            check = self.probe.TensorCheckpoints(directory, reference=False)
            check.record("final", (torch.arange(4), [torch.ones(2)]))
            self.assertTrue(check.finish()["passed"])
            missing = self.probe.TensorCheckpoints(directory, reference=False)
            self.assertFalse(missing.finish()["passed"])
            changed = self.probe.TensorCheckpoints(directory, reference=False)
            changed.record("final", (torch.arange(4), [torch.zeros(2)]))
            self.assertFalse(changed.finish()["passed"])

    def test_output_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                self.probe.new_output_dir(Path(tmp))

    def test_unimplemented_modes_rejected(self):
        for sp, batch in [(16, "off"), (1, "equal_length")]:
            with self.assertRaisesRegex(ValueError, "not implemented"):
                self.probe.validate_probe_mode(sp, batch)


if __name__ == "__main__":
    unittest.main()
