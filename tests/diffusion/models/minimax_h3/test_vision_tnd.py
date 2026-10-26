# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract for the opt-in NPU packed vision attention adapter."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_adapter():
    path = Path(__file__).parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text())
    nodes: list[ast.stmt] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tnd_attention"]
    assert nodes, "production TND adapter is missing"
    ns = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["tnd_attention"]


@pytest.mark.parametrize("lengths", [(2, 3, 1), (8160, 8160, 880, 660, 660, 7656)])
def test_tnd_adapter_preserves_axes_boundaries_and_scale(monkeypatch, lengths):
    total = sum(lengths)
    values = [torch.arange(total * 2 * 3).reshape(1, 2, total, 3) + offset for offset in (0, 10, 20)]
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(kwargs)
        for actual, original in zip((q, k, v), values):
            assert torch.equal(actual, original.squeeze(0).transpose(0, 1))
            assert actual.is_contiguous()
        return (v.clone(),)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=kernel))
    result = load_adapter()(*values, lengths)
    assert torch.equal(result, values[2].squeeze(0).transpose(0, 1))
    assert result.is_contiguous()
    assert len(calls) == 1
    ends = list(torch.tensor(lengths).cumsum(0).tolist())
    assert calls[0] == dict(
        head_num=2,
        input_layout="TND",
        scale=3**-0.5,
        keep_prob=1.0,
        actual_seq_qlen=ends,
        actual_seq_kvlen=ends,
        sparse_mode=0,
        inner_precise=0,
    )


@pytest.mark.parametrize(
    "enabled,verify,corrupt", [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
)
def test_attention_routing_and_verification(enabled, verify, corrupt):
    path = Path(__file__).parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "attention_sp8"
    )
    calls = []
    flags = {"VLLM_OMNI_H3_VISION_TND": enabled, "VLLM_OMNI_H3_VISION_TND_VERIFY": verify}

    def packed(q, k, v, lengths):
        calls.append("tnd")
        result = v.squeeze(0).transpose(0, 1).contiguous()
        return result + 1 if corrupt else result

    def segment(q, k, v):
        calls.append("segment")
        return v

    def reduce(status, **kwargs):
        calls.append("verify")

    ns = {
        "torch": torch,
        "vision_permutation_enabled": lambda name: flags[name],
        "tnd_attention": packed,
        "segment_attention": segment,
        "dist": SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX=0)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    transport = SimpleNamespace(
        plan=SimpleNamespace(segment_lengths=(2, 3)), group=None, to_heads=lambda x: x, to_tokens=lambda x: x
    )
    qkv = torch.arange(60, dtype=torch.float32).reshape(5, 12)
    attention = SimpleNamespace(qkv=lambda hidden: qkv, num_heads=2, head_dim=2, proj=lambda x: x)
    if corrupt:
        with pytest.raises(RuntimeError, match="changed bytes or strides"):
            ns["attention_sp8"](attention, torch.zeros(5, 4), (None, None), transport, lambda q, k, *args: (q, k))
        assert calls.count("verify") == 1
        return
    actual = ns["attention_sp8"](attention, torch.zeros(5, 4), (None, None), transport, lambda q, k, *args: (q, k))
    assert torch.equal(actual, qkv[:, 8:12])
    assert calls.count("tnd") == int(enabled)
    assert calls.count("segment") == (2 if not enabled or verify else 0)
    assert calls.count("verify") == int(verify)
