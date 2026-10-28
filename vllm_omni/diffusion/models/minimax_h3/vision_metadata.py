# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, model-local inference geometry. Never cache learned features."""

from collections import OrderedDict
from collections.abc import Hashable
from threading import RLock

import torch


class CpuTensorCache:
    def __init__(self, max_entries: int = 8, max_bytes: int = 8 * 1024**2):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[torch.Tensor, ...]] = OrderedDict()
        self._lock = RLock()
        self.nbytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: Hashable) -> tuple[torch.Tensor, ...] | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: Hashable, value: tuple[torch.Tensor, ...]) -> None:
        if any(t.device.type != "cpu" or t.requires_grad for t in value):
            raise ValueError("geometry cache accepts only CPU constants")
        size = sum(t.numel() * t.element_size() for t in value)
        if size > self.max_bytes or self.max_entries <= 0:
            return
        with self._lock:
            previous = self._entries.pop(key, ())
            self.nbytes -= sum(t.numel() * t.element_size() for t in previous)
            while self._entries and (len(self._entries) >= self.max_entries or self.nbytes + size > self.max_bytes):
                _, evicted = self._entries.popitem(last=False)
                self.nbytes -= sum(t.numel() * t.element_size() for t in evicted)
            self._entries[key] = value
            self.nbytes += size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.nbytes = 0

    def __getstate__(self):
        # Geometry is disposable and must not inflate model serialization.
        return {"max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.__init__(**state)
