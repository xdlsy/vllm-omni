# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental Ulysses correctness runner, launched with torchrun.

No serving integration or timing claims. A fresh full reference runs on rank0;
candidate ranks compute only their own tokens. First inequality stops all ranks.
Unexpected exceptions are bounded by the process-group and launcher timeouts.
"""

import argparse
import importlib.metadata
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vision_encoding_probe import (
    TensorCheckpoints,
    compare_tensor,
    digest,
    load_vision,
    new_output_dir,
    prepare,
    run_pass,
    tensor_summary,
    validate_inputs,
    verify_module_source,
    write_json,
)


class ShardChecks:
    def __init__(self, directory, device):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.device = device
        self.results = {}

    def check(self, name, candidate, indices=None):
        try:
            if name in self.results:
                raise ValueError("duplicate checkpoint")
            entry = self.manifest[name]
            reference = torch.load(self.directory / entry["file"], map_location="cpu", weights_only=True)
            if tensor_summary(reference)["sha256"] != entry["sha256"]:
                raise ValueError("reference digest changed")
            result = compare_tensor(reference if indices is None else reference[list(indices)], candidate)
        except Exception as error:
            result = {"passed": False, "error": str(error)}
        self.results[name] = result
        failure = torch.tensor([not result["passed"]], dtype=torch.int32, device=self.device)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if failure.item():
            raise RuntimeError(f"collective correctness gate failed at {name}")

    def finish(self):
        missing = sorted(set(self.manifest) - set(self.results))
        failed = missing or any(not point["passed"] for point in self.results.values())
        flag = torch.tensor([bool(failed)], dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise RuntimeError(f"missing or failed checkpoints: {missing}")


def segment_attention(q, k, v, *, short_min_heads=0):
    """Opt-in shape experiment; A5 short segments differed with fewer heads.

    Duplicate independent heads, never tokens, and discard duplicate outputs.
    This is not a general backend guarantee: every checkpoint still gates it.
    """
    heads = q.shape[1]
    if q.shape[2] <= 1024 and heads < short_min_heads:
        if short_min_heads % heads:
            raise ValueError("minimum heads must be divisible by local heads")
        inputs = [
            t.repeat(1, short_min_heads // heads, 1, 1).transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v)
        ]
        return F.scaled_dot_product_attention(*inputs, dropout_p=0.0, is_causal=False)[:, :heads]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def sharded_forward(model, pixels, grid, transport, checks, rotate, *, short_min_heads=0, batch=False, stats=None):
    """Keep hidden states sharded between blocks; global metadata is replicated."""
    plan = transport.plan
    ids = plan.rank_token_indices[transport.rank]
    merged_ids = plan.rank_merger_indices[transport.rank]
    select = torch.tensor(ids, dtype=torch.long, device=pixels.device)
    checks.check("input/0", pixels)
    checks.check("input/1", grid)
    hidden = model.patch_embed(pixels.index_select(0, select))
    checks.check("patch_embed", hidden, ids)
    hidden = hidden + model.fast_pos_embed_interpolate(grid).index_select(0, select)
    checks.check("block0_input", hidden, ids)
    rope = model.rot_pos_emb(grid).reshape(sum(plan.segment_lengths), -1)
    emb = torch.cat((rope, rope), dim=-1)
    cos, sin = emb.cos().index_select(0, select), emb.sin().index_select(0, select)
    checks.check("position_embeddings/0", cos, ids)
    checks.check("position_embeddings/1", sin, ids)
    cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    checks.check("cu_seqlens", F.pad(cu, (1, 0), value=0))
    deepstack = []
    for number, block in enumerate(model.blocks):
        prefix = f"blocks.{number}"
        attention = block.attn
        qkv = attention.qkv(block.norm1(hidden))
        checks.check(prefix + ".attn.qkv", qkv, ids)
        q, k, v = qkv.reshape(len(ids), 3, attention.num_heads, attention.head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = rotate(q, k, cos, sin)
        q, k, v = [transport.to_heads(tensor).transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)]
        splits = [torch.split(tensor, plan.segment_lengths, dim=2) for tensor in (q, k, v)]

        def attend(q, k, v):
            if stats is not None:
                stats["sdpa_calls"] = stats.get("sdpa_calls", 0) + 1
                stats["batched_calls"] = stats.get("batched_calls", 0) + (q.shape[0] > 1)
            return segment_attention(q, k, v, short_min_heads=short_min_heads)

        if batch:
            from vllm_omni.diffusion.models.minimax_h3 import vision_batch

            verify_module_source(vision_batch, "vllm_omni/diffusion/models/minimax_h3/vision_batch.py")
            outputs = vision_batch.run_grouped_attention(*splits, attention=attend)
        else:
            outputs = [attend(qi, ki, vi) for qi, ki, vi in zip(*splits)]
        full = torch.cat(outputs, dim=2).squeeze(0).transpose(0, 1).contiguous()
        local = transport.to_tokens(full).reshape(len(ids), model.config.hidden_size)
        projected = attention.proj(local)
        checks.check(prefix + ".attn", projected, ids)
        hidden = hidden + projected
        mlp = block.mlp(block.norm2(hidden))
        checks.check(prefix + ".mlp", mlp, ids)
        hidden = hidden + mlp
        checks.check(prefix, hidden, ids)
        if number in model.deepstack_visual_indexes:
            deep_index = model.deepstack_visual_indexes.index(number)
            output = model.deepstack_merger_list[deep_index](hidden)
            checks.check(f"deepstack_merger_list.{deep_index}", output, merged_ids)
            deepstack.append(output)
    final = model.merger(hidden)
    checks.check("merger", final, merged_ids)
    checks.check("final/0", final, merged_ids)
    for index, output in enumerate(deepstack):
        checks.check(f"final/1/{index}", output, merged_ids)
    return final, deepstack


def transport_smoke(parallel, size, device):
    plan = parallel.build_vision_plan((4, 0, 20, 64, 12, 8), size)
    transport = parallel.VisionAllToAll(plan)
    x = torch.arange(108 * 16 * 2, dtype=torch.float32, device=device).reshape(108, 16, 2)
    ids = torch.tensor(plan.rank_token_indices[transport.rank], dtype=torch.long, device=device)
    local = x.index_select(0, ids)
    passed = True
    for _ in range(4):
        heads = transport.to_heads(local)
        expected = x[:, transport.rank * (16 // size) : (transport.rank + 1) * (16 // size)]
        restored = transport.to_tokens(heads)
        passed = passed and torch.equal(heads, expected) and torch.equal(restored, local)
    failed = torch.tensor([not passed], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError("transport smoke failed")
    return {"passed": True, "iterations": 4, "a2a_calls": transport.calls, "sent_bytes": transport.sent_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "model-root", "refs-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sp-size", type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--short-sdpa-min-heads", type=int, choices=(0, 8), default=0)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != args.sp_size:
        raise ValueError("SP must equal actual launched world; no simulated ranks")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    import torch_npu  # noqa: F401

    from vllm_omni.diffusion.models.minimax_h3 import encoder, vision_parallel

    verify_module_source(vision_parallel, "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=900))
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": args.sp_size,
        "passed": False,
        "scope": "isolated vision; not serving or performance",
        "batch": args.batch,
        "attention_stats": {},
        "short_sdpa_min_heads": args.short_sdpa_min_heads,
        "short_sdpa_max_tokens": 1024,
    }
    transport, checks = None, None
    owned_output = False
    try:
        if rank == 0:
            new_output_dir(args.output)
        dist.barrier()
        owned_output = True
        case = validate_inputs(args.case, args.refs_dir)
        config, inputs, sizes = prepare(args, case)
        lengths = tuple(h * w for t, h, w in inputs["image_grid_thw"].tolist() for _ in range(t))
        plan = vision_parallel.build_vision_plan(
            lengths, args.sp_size, num_heads=config.num_heads, merge_size=config.spatial_merge_size
        )
        report["transport_smoke"] = transport_smoke(vision_parallel, args.sp_size, device)
        model, weights = load_vision(args.model_root, config)
        model.to(device)
        pixels = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        grid = inputs["image_grid_thw"].to(device)
        report.update(
            {
                "local_tokens": len(plan.rank_token_indices[rank]),
                "heads_per_rank": plan.heads_per_rank,
                "token_indices": plan.rank_token_indices[rank],
                "merger_indices": plan.rank_merger_indices[rank],
                "case_sha256": digest(args.case),
                "weights": weights,
                "torch": torch.__version__,
                "source": {
                    str(p): digest(p) for p in (Path(__file__), Path(encoder.__file__), Path(vision_parallel.__file__))
                },
            }
        )
        source_root = Path(__file__).resolve().parents[2]
        report["source"].update(
            {
                str(source_root / name): digest(source_root / name)
                for name in (
                    "benchmarks/minimax_h3/vision_encoding_probe.py",
                    "vllm_omni/diffusion/models/minimax_h3/vision_batch.py",
                    "vllm_omni/model_executor/models/minimax_h3/preprocessing.py",
                    "vllm_omni/model_executor/models/minimax_h3/encoder_processing.py",
                )
            }
        )
        report.update(
            grid=grid.cpu().tolist(),
            segment_lengths=lengths,
            prepared_image_sizes=sizes,
            model_config_sha256=digest(args.model_root / "text_encoder/config.json"),
            processor_files={str(p): digest(p) for p in sorted((args.model_root / "processor").glob("*.json"))},
            packages={name: importlib.metadata.version(name) for name in ("torch-npu", "vllm", "transformers")},
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        )
        if rank == 0:
            result = run_pass(model, pixels, grid, TensorCheckpoints(args.output / "reference", reference=True))
            if not result["passed"]:
                raise RuntimeError("reference contains nonfinite values")
        dist.barrier()
        transport = vision_parallel.VisionAllToAll(plan)
        checks = ShardChecks(args.output / "reference", device)
        with torch.inference_mode():
            sharded_forward(
                model,
                pixels,
                grid,
                transport,
                checks,
                encoder._apply_rotary_pos_emb_vision,
                short_min_heads=args.short_sdpa_min_heads,
                batch=args.batch,
                stats=report["attention_stats"],
            )
        checks.finish()
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if transport is not None:
            report.update(a2a_calls=transport.calls, sent_bytes=transport.sent_bytes)
        if checks is not None:
            report["points"] = checks.results
        if owned_output:
            write_json(args.output / f"rank-{rank}.json", report)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
