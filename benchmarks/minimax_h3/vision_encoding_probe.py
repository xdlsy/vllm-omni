# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated H3 vision correctness probe. Diagnostic hooks invalidate timings.

Only the original SP1/batch-off path is executable. Inspect is CPU-only;
correctness requires explicit device selection and an externally idle device.
This is not HTTP, multi-rank, Turbo/DiT or end-to-end acceptance.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import torch


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_output_dir(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_probe_mode(sp, batch):
    if sp != 1 or batch != "off":
        raise ValueError("SP/Batch execution not implemented; only reference SP1/off is available")


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def tensor_summary(tensor):
    raw = tensor_bytes(tensor)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def compare_tensor(reference, candidate):
    reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
    result = {"reference": tensor_summary(reference), "candidate": tensor_summary(candidate), "passed": False}
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        result["reason"] = "shape_or_dtype"
        return result
    equal_bytes = tensor_bytes(reference) == tensor_bytes(candidate)
    result["torch_equal"] = torch.equal(reference, candidate)
    result["passed"] = bool(equal_bytes.all()) and result["reference"]["finite"] and result["candidate"]["finite"]
    if reference.numel():
        delta = (reference.double() - candidate.double()).abs()
        result["max_abs_diff"] = delta.max().item() if bool(torch.isfinite(delta).all()) else None
    else:
        result["max_abs_diff"] = 0.0
    if not bool(equal_bytes.all()):
        flat = int((~equal_bytes).nonzero()[0].item()) // reference.element_size()
        coords = []
        for size in reversed(reference.shape):
            coords.append(flat % size)
            flat //= size
        result["first_mismatch"] = list(reversed(coords))
    return result


def flatten_tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, (list, tuple)):
        return {
            key: tensor for i, item in enumerate(value) for key, tensor in flatten_tensors(item, f"{path}/{i}").items()
        }
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


class TensorCheckpoints:
    """Persist one reference tensor per point; compare without retaining a tower."""

    def __init__(self, directory, *, reference):
        self.directory, self.reference = Path(directory), reference
        self.seen, self.results = set(), {}
        if reference:
            new_output_dir(self.directory)
            self.manifest = {}
        else:
            self.manifest = json.loads((self.directory / "manifest.json").read_text())

    def record(self, name, value):
        for suffix, tensor in flatten_tensors(value).items():
            key = name + suffix
            if key in self.seen:
                raise ValueError(f"duplicate checkpoint: {key}")
            self.seen.add(key)
            if self.reference:
                filename = f"{len(self.manifest):05d}.pt"
                cpu = tensor.detach().cpu().contiguous()
                torch.save(cpu, self.directory / filename)
                self.manifest[key] = {"file": filename, **tensor_summary(cpu)}
            elif key not in self.manifest:
                self.results[key] = {"passed": False, "reason": "unexpected_point"}
            else:
                entry = self.manifest[key]
                reference = torch.load(self.directory / entry["file"], map_location="cpu", weights_only=True)
                if tensor_summary(reference) != {k: entry[k] for k in ("shape", "dtype", "sha256", "finite")}:
                    raise ValueError(f"reference checkpoint integrity failure: {key}")
                self.results[key] = compare_tensor(reference, tensor)

    def finish(self):
        if self.reference:
            write_json(self.directory / "manifest.json", self.manifest)
            return {"passed": bool(self.manifest) and all(p["finite"] for p in self.manifest.values())}
        missing = sorted(set(self.manifest) - self.seen)
        return {
            "passed": bool(self.results) and not missing and all(p["passed"] for p in self.results.values()),
            "missing": missing,
            "points": self.results,
        }


def validate_inputs(case_path, refs_dir):
    case = json.loads(case_path.read_text(encoding="utf-8"))
    manifest = json.loads((case_path.parent / "manifest.sha256.json").read_text())
    if case_path.name not in manifest or "prompt.txt" not in manifest:
        raise ValueError("baseline manifest must cover the actual case and prompt")
    for name, expected in manifest.items():
        if digest(case_path.parent / name) != expected:
            raise ValueError(f"frozen baseline changed: {name}")
    config = case["configuration"]
    if (
        case.get("baseline_id") != "H3-REF2VA-TURBO-8CARD-v1"
        or config["task"] != "ref2va"
        or config["refs"] != [f"ref-{i}.png" for i in range(1, 7)]
    ):
        raise ValueError("this probe requires the frozen six-image Ref2VA case")
    for name in config["refs"]:
        if digest(refs_dir / name) != case["input_reference_sha256"][name]:
            raise ValueError(f"reference image changed: {name}")
    if digest(case_path.parent / "prompt.txt") != config["prompt_sha256"]:
        raise ValueError("prompt changed")
    return case


def verify_module_source(module, relative_path):
    expected = Path(__file__).resolve().parents[2] / relative_path
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError(f"imported source does not match probe checkout: {module.__file__}")
    return {"path": str(expected), "sha256": digest(expected)}


def prepare(args, case):
    from transformers import Qwen3VLConfig, Qwen3VLProcessor

    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing, preprocessing

    for module in (encoder_processing, preprocessing):
        verify_module_source(module, module.__name__.replace(".", "/") + ".py")

    config = Qwen3VLConfig.from_pretrained(args.model_root / "text_encoder", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(args.model_root / "processor", local_files_only=True)
    images = preprocessing.load_minimax_h3_images([str(args.refs_dir / n) for n in case["configuration"]["refs"]])
    prepared = encoder_processing._prepare_encoder_images("ref2va", images, height=768, width=1344, guided=False)
    encoded = processor.image_processor(images=prepared, return_tensors="pt")
    return config.vision_config, encoded, [list(image.size) for image in prepared]


def load_vision(model_root, config):
    from safetensors import safe_open

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    verify_module_source(encoder, "vllm_omni/diffusion/models/minimax_h3/encoder.py")
    model = encoder.MiniMaxH3Qwen3VLVisionModel(config).to(dtype=torch.bfloat16).eval()
    state, sources = {}, {}
    for path in sorted((model_root / "text_encoder").glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = [key for key in reader.keys() if key.startswith("model.visual.")]
            if keys:
                sources[path.name] = digest(path)
            for key in keys:
                short = key.removeprefix("model.visual.")
                if short in state:
                    raise ValueError(f"duplicate weight {key}")
                state[short] = reader.get_tensor(key)
    model.load_state_dict(state, strict=True)
    return model, sources


def run_pass(model, pixels, grid, checkpoints):
    handles = []
    try:
        checkpoints.record("input", (pixels, grid))

        def block_inputs(_module, args, kwargs):
            checkpoints.record("block0_input", args[0])
            checkpoints.record("cu_seqlens", kwargs["cu_seqlens"])
            checkpoints.record("position_embeddings", kwargs["position_embeddings"])

        handles.append(model.blocks[0].register_forward_pre_hook(block_inputs, with_kwargs=True))
        for name, module in model.named_modules():
            if (
                name == "patch_embed"
                or name == "merger"
                or name.startswith("deepstack_merger_list.")
                and name.count(".") == 1
                or name.startswith("blocks.")
                and (name.count(".") == 1 or name.endswith((".attn", ".attn.qkv", ".mlp")))
            ):

                def hook(_module, _inputs, output, point=name):
                    checkpoints.record(point, output)

                handles.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            checkpoints.record("final", model(pixels, grid))
    finally:
        for handle in handles:
            handle.remove()
    return checkpoints.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "model-root", "refs-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--platform", choices=("A2", "A5"), required=True)
    parser.add_argument("--mode", choices=("inspect", "correctness"), default="inspect")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--batch-mode", default="off")
    args = parser.parse_args()
    validate_probe_mode(args.sp_size, args.batch_mode)
    case = validate_inputs(args.case, args.refs_dir)
    out = new_output_dir(args.output)
    try:
        config, inputs, sizes = prepare(args, case)
        grid = inputs["image_grid_thw"]
        lengths = [h * w for t, h, w in grid.tolist() for _ in range(t)]
        shape = {
            "vision_config": config.to_dict(),
            "grid_thw": grid.tolist(),
            "segment_lengths": lengths,
            "prepared_image_sizes": sizes,
            "pixel_values": tensor_summary(inputs["pixel_values"]),
            "merger_tokens": sum(lengths) // config.spatial_merge_size**2,
            "head_dim": config.hidden_size // config.num_heads,
            "sp16_heads_divisible": config.num_heads % 16 == 0,
        }
        write_json(out / "shape-contract.json", shape)
        source_root = Path(__file__).resolve().parents[2]
        source_files = [
            "benchmarks/minimax_h3/vision_encoding_probe.py",
            "vllm_omni/diffusion/models/minimax_h3/encoder.py",
            "vllm_omni/model_executor/models/minimax_h3/encoder_processing.py",
            "vllm_omni/model_executor/models/minimax_h3/preprocessing.py",
        ]
        provenance = {
            "baseline_id": case["baseline_id"],
            "case_sha256": digest(args.case),
            "platform_label": args.platform,
            "device": args.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "packages": {n: importlib.metadata.version(n) for n in ("transformers", "vllm")},
            "source": {n: digest(source_root / n) for n in source_files},
            "model_config_sha256": digest(args.model_root / "text_encoder/config.json"),
            "processor_files": {
                str(p.relative_to(args.model_root)): digest(p)
                for p in sorted((args.model_root / "processor").glob("*.json"))
            },
            "scope": "isolated vision; Turbo belongs to downstream DiT and is not loaded",
        }
        write_json(out / "source-manifest.json", provenance)
        if args.mode == "correctness":
            if not args.device.startswith("npu:"):
                raise ValueError("real correctness requires explicit npu:<index>")
            import torch_npu  # noqa: F401

            # Full per-layer dumps need space; refuse before allocating device tensors.
            estimate = inputs["pixel_values"].shape[0] * config.hidden_size * 2 * config.depth * 7
            if shutil.disk_usage(out).free < estimate * 2:
                raise ValueError("insufficient space for reference checkpoint dumps")
            model, weights = load_vision(args.model_root, config)
            provenance["weight_shards_sha256"] = weights
            write_json(out / "source-manifest.json", provenance)
            torch.manual_seed(case["configuration"]["seed"])
            model.to(args.device)
            pixels = inputs["pixel_values"].to(args.device, dtype=torch.bfloat16)
            grid = grid.to(args.device)
            first = run_pass(model, pixels, grid, TensorCheckpoints(out / "reference", reference=True))
            second = run_pass(model, pixels, grid, TensorCheckpoints(out / "reference", reference=False))
            second["passed"] = first["passed"] and second["passed"]
            second["scope"] = "same-process SP1/off repeat; not cross-configuration or E2E acceptance"
            write_json(out / "correctness.json", second)
            if not second["passed"]:
                raise RuntimeError("bitwise repeat failed; see correctness.json")
    except Exception as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
