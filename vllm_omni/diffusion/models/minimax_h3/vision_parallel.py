# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU layout contracts for a future H3 vision stage; no runtime collectives."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionShardPlan:
    segment_lengths: tuple[int, ...]
    rank_token_indices: tuple[tuple[int, ...], ...]
    rank_merger_indices: tuple[tuple[int, ...], ...]
    heads_per_rank: int


def build_vision_plan(lengths, sp_size, *, num_heads=16, merge_size=2):
    """Partition each segment into contiguous whole spatial merger groups.

    Uneven/empty shards are explicit. This does not imply that padded attention
    or smaller projection GEMMs will be bitwise equivalent on a device.
    """
    if sp_size not in (1, 2, 4, 8, 16) or num_heads <= 0 or num_heads % sp_size:
        raise ValueError("SP must divide the positive head count and be one of 1/2/4/8/16")
    if not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("merge_size must be a positive integer")
    unit = merge_size**2
    lengths = tuple(lengths)
    if any(not isinstance(n, int) or n < 0 or n % unit for n in lengths):
        raise ValueError("segment lengths must contain whole nonnegative merger groups")
    tokens = [[] for _ in range(sp_size)]
    mergers = [[] for _ in range(sp_size)]
    offset = 0
    for length in lengths:
        groups, remainder = divmod(length // unit, sp_size)
        start = offset
        for rank in range(sp_size):
            stop = start + unit * (groups + (rank < remainder))
            tokens[rank].extend(range(start, stop))
            mergers[rank].extend(range(start // unit, stop // unit))
            start = stop
        offset += length
    return VisionShardPlan(lengths, tuple(map(tuple, tokens)), tuple(map(tuple, mergers)), num_heads // sp_size)


def validate_stage_layout(world_size, vision_ranks, downstream_ranks, sp_size, num_heads, *, data_parallel_size=1):
    """Validate a proposed layout, not the current serving runtime's support."""
    build_vision_plan((), sp_size, num_heads=num_heads)
    if world_size < sp_size or any(r < 0 or r >= world_size for r in (*vision_ranks, *downstream_ranks)):
        raise ValueError("vision/downstream rank is outside the world")
    if data_parallel_size != 1:
        raise ValueError("implicit DP replication is not the independent vision stage")
    if tuple(downstream_ranks) != tuple(range(8)):
        raise ValueError("the frozen downstream stage must use ranks 0..7")
    if len(set(vision_ranks)) != sp_size or len(vision_ranks) != sp_size:
        raise ValueError("vision participants must be distinct and match SP")
