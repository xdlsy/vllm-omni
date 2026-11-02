# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scoped, opt-in wall-clock timing shared by a pipeline and its components."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar


class StageTimer:
    """Accumulate exclusive host wall-clock seconds without device synchronization.

    The caller owns enabling and reporting. Pending device work is not awaited.
    Component labels apply only to the selected component; unrelated model
    swaps are intentionally excluded. No distributed collectives are added.
    """

    def __init__(
        self,
        *,
        component: object | None = None,
        component_stages: Mapping[str, str] | None = None,
    ) -> None:
        self.component = component
        self.component_stages = component_stages or {}
        self.durations: dict[str, float] = {}
        self._children: list[float] = []

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active_timer.set(self)
        try:
            yield
        finally:
            _active_timer.reset(token)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        self._children.append(0.0)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            children = self._children.pop()
            self.durations[name] = self.durations.get(name, 0.0) + elapsed - children
            if self._children:
                self._children[-1] += elapsed


_active_timer: ContextVar[StageTimer | None] = ContextVar("diffusion_stage_timer", default=None)


@contextmanager
def timed_stage(name: str, *, component: object | None = None) -> Iterator[None]:
    """Measure only within an active scope; otherwise do no timing or syncing."""
    timer = _active_timer.get()
    if timer is None or (component is not None and component is not timer.component):
        yield
        return
    if component is not None:
        label = timer.component_stages.get(name)
        if label is None:
            yield
            return
        name = label
    with timer.measure(name):
        yield
