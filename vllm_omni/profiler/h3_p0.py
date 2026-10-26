# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in, bounded Host spans for H3's pre-DiT path. Never synchronizes devices."""

import inspect
import json
import logging
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, copy_context
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

_state = ContextVar("h3_p0_state", default=None)
_parent = ContextVar("h3_p0_parent", default=None)
_labels = ContextVar("h3_p0_labels", default={})
_write_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def metadata(value):
    """Return storage-free metadata, including for unsupported tensor layouts."""
    try:
        return _metadata(value)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {"type": type(value).__name__, "metadata_unavailable": True}


def _metadata(value):
    """Describe tensors without reading their contents or retaining their storage."""
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        result = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if hasattr(value, "stride"):
            result["stride"] = list(value.stride())
            result["contiguous"] = value.is_contiguous()
            result["logical_bytes"] = value.numel() * value.element_size()
        return result
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "items": [metadata(v) for v in value[:16]]}
    if isinstance(value, dict):
        return {
            str(k): {"tensor": metadata(v), "cpu_grid": grid_metadata(v)}
            if str(k).endswith("grid_thw")
            else metadata(v)
            for k, v in list(value.items())[:32]
        }
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {"size": list(value.size), "mode": value.mode}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray, str)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def annotate(**fields):
    """Attach small, explicitly selected scalar fields to the current request."""
    state = _state.get()
    if state is not None and state.active and state.capture:
        state.fields.update(fields)


def grid_metadata(grid):
    """Read only the small CPU grid already produced by the processor."""
    if _state.get() is None or str(getattr(grid, "device", "")) != "cpu":
        return None
    if len(grid.shape) == 2 and grid.shape[0] <= 32 and grid.shape[1] == 3:
        return grid.tolist()
    return None


def _trace(name):
    torch = sys.modules.get("torch")
    profiler = getattr(torch, "profiler", None)
    return profiler.record_function(f"H3_STAGE/{name}") if profiler else nullcontext()


@contextmanager
def stage(name, *, trace=True, **fields):
    state = _state.get()
    if state is None or not state.active or not state.capture:
        yield None
        return
    with state.lock:
        state.sequence += 1
        state.counts[name] = state.counts.get(name, 0) + 1
        span_id = f"{state.identity}:{state.sequence}"
        occurrence = state.counts[name]
    record = {
        "schema_version": 1,
        "stage": name,
        "span_id": span_id,
        "parent_span_id": _parent.get(),
        "begin_ns": time.perf_counter_ns(),
        "tid": threading.get_ident(),
        "status": "ok",
        "occurrence": occurrence,
        **_labels.get(),
        **fields,
    }
    token = _parent.set(span_id)
    labels_token = _labels.set({**_labels.get(), **fields})
    try:
        with _trace(name) if trace else nullcontext():
            yield record
    except BaseException as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        record["end_ns"] = time.perf_counter_ns()
        _parent.reset(token)
        _labels.reset(labels_token)
        with state.lock:
            if len(state.events) < state.max_events:
                state.events.append(record)
            else:
                state.dropped += 1


@contextmanager
def request(request_id=None, *, max_events=8192, force=False, **fields):
    """Own one buffer per API task or worker invocation and flush on exit."""
    directory = os.environ.get("VLLM_OMNI_H3_P0_PROFILE_DIR")
    previous = _state.get()
    if not directory or (previous is not None and previous.active and not force):
        yield
        return
    identity = uuid.uuid4().hex
    state = SimpleNamespace(
        identity=identity,
        sequence=0,
        fields={
            "run_id": os.environ.get("VLLM_OMNI_H3_P0_RUN_ID", "unspecified"),
            "request_id": request_id,
            "trace_id": identity,
            "parent_trace_id": previous.identity if previous is not None else None,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "global_rank": os.environ.get("RANK", "unknown"),
            "local_rank": os.environ.get("LOCAL_RANK", "unknown"),
            "clock_wall_ns": time.time_ns(),
            "clock_monotonic_ns": time.perf_counter_ns(),
            **fields,
        },
        events=[],
        max_events=max_events,
        dropped=0,
        capture=True,
        active=True,
        counts={},
        collectives={},
        lock=threading.Lock(),
    )
    token = _state.set(state)
    parent_token = _parent.set(None)
    labels_token = _labels.set({})
    try:
        with stage("profile/request", trace=False):
            yield
    finally:
        state.active = False
        _parent.reset(parent_token)
        _labels.reset(labels_token)
        _state.reset(token)
        summary = {"stage": "profile/summary", "dropped_events": state.dropped, "schema_version": 1}
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            records = [{**state.fields, **event} for event in [*state.events, summary]]
            content = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records)
            with _write_lock, (path / f"h3-p0-{os.getpid()}.jsonl").open("a", encoding="utf-8") as output:
                output.write(content)
        except (OSError, TypeError, ValueError):
            _logger.warning("Could not write optional H3 P0 profiling data", exc_info=False)


def profile(name):
    """Wrap a coarse function boundary; disabled requests retain no metadata."""

    def decorate(function):
        if inspect.iscoroutinefunction(function):

            @wraps(function)
            async def async_wrapper(*args, **kwargs):
                with stage(name, trace=False) as record:
                    if record is not None:
                        record["inputs"] = metadata(args)
                        record["kwargs"] = metadata(kwargs)
                    result = await function(*args, **kwargs)
                    if record is not None:
                        record["output"] = metadata(result)
                    return result

            return async_wrapper

        @wraps(function)
        def wrapper(*args, **kwargs):
            return call(name, function, *args, **kwargs)

        return wrapper

    return decorate


def call(name, function, *args, **kwargs):
    """Time one existing call without changing its arguments or result."""
    with stage(name) as record:
        if record is not None:
            record["inputs"] = metadata(args)
            record["kwargs"] = metadata(kwargs)
            record["function"] = getattr(function, "__qualname__", type(function).__qualname__)
            record["module"] = getattr(function, "__module__", type(function).__module__)
            owner = getattr(function, "__self__", None)
            if owner is not None:
                record["bound_input"] = metadata(owner)
            collective_names = ("broadcast", "broadcast_object_list", "all_gather_object")
            if "group" in kwargs and getattr(function, "__name__", "") in collective_names:
                group_id = "encoder_tp" if name.startswith("qwen/") else "dit_world"
                state = _state.get()
                state.collectives[group_id] = state.collectives.get(group_id, 0) + 1
                record["group_id"] = group_id
                record["collective_seq"] = state.collectives[group_id]
        result = function(*args, **kwargs)
        if record is not None:
            record["output"] = metadata(result)
        return result


def first_dit(**fields):
    """Mark the logical first DiT call and stop subsequent P0 child capture."""
    with stage("dit/first_model_enter", **fields):
        pass
    state = _state.get()
    if state is not None:
        state.capture = False


def submit(name, executor, function, *args, **kwargs):
    """Carry the request trace into an existing background task, without waiting."""
    state = _state.get()
    if state is None or not state.active or not state.capture:
        return executor.submit(function, *args, **kwargs)
    context = copy_context()
    return executor.submit(context.run, call, name, function, *args, **kwargs)


def event(name, **fields):
    with stage(name, **fields):
        pass


class _ProfiledScope:
    def __init__(self, name, manager):
        self.name = name
        self.manager = manager

    def __enter__(self):
        with stage(f"{self.name}/enter"):
            return self.manager.__enter__()

    def __exit__(self, *exc):
        with stage(f"{self.name}/exit"):
            return self.manager.__exit__(*exc)


def scope(name, manager):
    """Time context entry/exit while preserving exception and suppression semantics."""
    return _ProfiledScope(name, manager)
