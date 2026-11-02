# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run isolated CPU lifecycle tests without installing vLLM.

Only logging, platform streams/events and package initialization are stubbed.
Tensor storage/rebinding/copy and module forward use real PyTorch. This runner
cannot validate accelerator ordering, pinning, model kernels or performance.
"""

import logging
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path else []
    sys.modules[name] = module
    return module


def main():
    package("vllm")
    logger = package("vllm.logger")
    logger.init_logger = logging.getLogger
    package("vllm_omni", ROOT / "vllm_omni")
    package("vllm_omni.diffusion", ROOT / "vllm_omni/diffusion")
    package("vllm_omni.diffusion.offloader", ROOT / "vllm_omni/diffusion/offloader")
    platform = package("vllm_omni.platforms")
    platform.current_omni_platform = types.SimpleNamespace(
        Stream=Mock,
        Event=Mock,
        current_stream=Mock,
        stream=lambda _: nullcontext(),
        synchronize=Mock(),
        empty_cache=Mock(),
        get_device_memory=lambda _: (80, 100),
    )
    import pytest

    class CpuFixtures:
        @pytest.fixture
        def mocker(self):
            patchers = []

            def patch_object(*args, **kwargs):
                patcher = patch.object(*args, **kwargs)
                patchers.append(patcher)
                return patcher.start()

            yield types.SimpleNamespace(Mock=Mock, patch=types.SimpleNamespace(object=patch_object))
            for patcher in reversed(patchers):
                patcher.stop()

    return pytest.main(
        [
            "--noconftest",
            "-o",
            "addopts=",
            "-q",
            str(ROOT / "tests/diffusion/offloader/test_encoder_residency.py"),
            str(ROOT / "tests/diffusion/offloader/test_module_residency.py"),
            *sys.argv[1:],
        ],
        plugins=[CpuFixtures()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
