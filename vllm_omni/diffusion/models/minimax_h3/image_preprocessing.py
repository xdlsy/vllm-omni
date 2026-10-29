# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in CPU patch packing for the Qwen image processor used by H3."""

from __future__ import annotations

import copy
import inspect
import os
from functools import partial
from typing import Any

import torch


def _packed_preprocess(processor, images, *, reference, **kwargs):
    from transformers.image_processing_utils import BatchFeature
    from transformers.image_transforms import group_images_by_shape, reorder_images
    from transformers.image_utils import SizeDict
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if any(not isinstance(image, torch.Tensor) or image.device.type != "cpu" for image in images):
        return reference(images=images, **kwargs)
    patch = kwargs["patch_size"]
    merge = kwargs["merge_size"]
    temporal = kwargs["temporal_patch_size"]
    grouping = kwargs["disable_grouping"]
    size = kwargs["size"]
    groups, order = group_images_by_shape(images, disable_grouping=grouping)
    resize_with_factor = "factor" in inspect.signature(processor.resize).parameters
    resized = {}
    for shape, batch in groups.items():
        if kwargs["do_resize"]:
            if resize_with_factor:
                batch = processor.resize(batch, size=size, resample=kwargs["resample"], factor=patch * merge)
            else:
                height, width = smart_resize(
                    *batch.shape[-2:],
                    factor=patch * merge,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                batch = processor.resize(batch, size=SizeDict(height=height, width=width), resample=kwargs["resample"])
        resized[shape] = batch
    images = reorder_images(resized, order)
    # The public processor still owns input conversion and all arithmetic.
    # Only the strided-to-contiguous packing destination changes here.
    groups, order = group_images_by_shape(images, disable_grouping=grouping)
    grids = [(image.shape[-2] // patch, image.shape[-1] // patch) for image in images]
    offsets = [0]
    for height, width in grids:
        offsets.append(offsets[-1] + height * width)
    pixels = torch.empty(
        (offsets[-1], images[0].shape[0] * temporal * patch * patch), dtype=torch.bfloat16, device="cpu"
    )
    for shape, batch in groups.items():
        normalized = processor.rescale_and_normalize(
            batch,
            kwargs["do_rescale"],
            kwargs["rescale_factor"],
            kwargs["do_normalize"],
            kwargs["image_mean"],
            kwargs["image_std"],
        )
        count, channels, height, width = normalized.shape
        gh, gw = height // patch, width // patch
        view = normalized.reshape(count, channels, gh // merge, merge, patch, gw // merge, merge, patch)
        view = view.permute(0, 2, 5, 3, 6, 1, 4, 7).unsqueeze(6)
        for index, (group_shape, group_index) in order.items():
            if group_shape != shape:
                continue
            destination = pixels[offsets[index] : offsets[index + 1]].view(
                gh // merge,
                gw // merge,
                merge,
                merge,
                channels,
                temporal,
                patch,
                patch,
            )
            destination.copy_(view[group_index].expand_as(destination))
    grid = torch.tensor([[1, height, width] for height, width in grids], dtype=torch.long, device="cpu")
    return BatchFeature(data={"pixel_values": pixels, "image_grid_thw": grid}, tensor_type=kwargs["return_tensors"])


def preprocess_images(processor: Any, images: list[Any]):
    """Preserve the default/custom processor; opt in to request-owned BF16 packing."""
    if os.environ.get("VLLM_OMNI_H3_IMAGE_PREPROCESS", "0") != "1":
        return processor(images=images, return_tensors="pt")
    try:
        from transformers.image_processing_backends import TorchvisionBackend
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    except ImportError:
        return processor(images=images, return_tensors="pt")
    if (
        type(processor) is not Qwen2VLImageProcessor
        or not isinstance(processor, TorchvisionBackend)
        or getattr(processor._preprocess, "__func__", None) is not Qwen2VLImageProcessor._preprocess
        or not {"disable_grouping", "patch_size", "merge_size", "temporal_patch_size"}.issubset(
            inspect.signature(processor._preprocess).parameters
        )
        or not images
    ):
        return processor(images=images, return_tensors="pt")
    optimized = copy.copy(processor)
    optimized._preprocess = partial(_packed_preprocess, optimized, reference=processor._preprocess)
    result = optimized(images=images, return_tensors="pt")
    if os.environ.get("VLLM_OMNI_H3_IMAGE_PREPROCESS_VERIFY", "0") == "1":
        reference = processor(images=images, return_tensors="pt")
        for key in ("pixel_values", "image_grid_thw"):
            actual = result[key]
            expected = reference[key].to(actual.dtype).contiguous()
            if actual.shape != expected.shape or not torch.equal(
                actual.contiguous().view(torch.uint8), expected.view(torch.uint8)
            ):
                raise AssertionError(f"MiniMax H3 image preprocessing is not byte-exact: {key}")
    return result
