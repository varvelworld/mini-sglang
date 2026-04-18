"""
Test DecodeManager.abort_req O(1) dict lookup refactor.

Verifies that running_reqs (Dict[int, Req]) has correct semantics for
abort_req, remove_req, filter_reqs, inflight_tokens, and schedule_next_batch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import pytest
from unittest.mock import MagicMock

from minisgl.core import Req, SamplingParams

# 直接加载 decode.py，绕开 scheduler/__init__.py（会拉入 zmq/msgpack 等重依赖）
_decode_path = Path(__file__).parents[2] / "python/minisgl/scheduler/decode.py"
_spec = importlib.util.spec_from_file_location("minisgl.scheduler.decode", _decode_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
sys.modules["minisgl.scheduler.decode"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore
DecodeManager = _mod.DecodeManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(uid: int, remain: int = 5) -> Req:
    """Construct a minimal Req that does not require GPU.

    Args:
        uid:    unique request id
        remain: output_len — controls can_decode (True when remain > 0)
    """
    return Req(
        input_ids=torch.tensor([1], dtype=torch.int32),  # must be CPU tensor
        table_idx=uid,
        cached_len=0,
        output_len=remain,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=MagicMock(),  # DecodeManager never touches cache_handle
    )


def make_manager(*reqs: Req) -> DecodeManager:
    """Create a DecodeManager pre-populated with the given reqs."""
    dm = DecodeManager(page_size=4)
    dm.running_reqs = {req.uid: req for req in reqs}
    return dm


# ---------------------------------------------------------------------------
# abort_req
# ---------------------------------------------------------------------------

def test_abort_req_hit():
    """命中时返回对应 Req，dict 里不再有该 uid。"""
    r0, r1, r2 = make_req(0), make_req(1), make_req(2)
    dm = make_manager(r0, r1, r2)

    result = dm.abort_req(1)

    assert result is r1, "should return the aborted Req"
    assert 1 not in dm.running_reqs, "uid 1 should be removed"
    assert len(dm.running_reqs) == 2, "remaining count should be 2"
    assert 0 in dm.running_reqs and 2 in dm.running_reqs


def test_abort_req_miss():
    """未命中时返回 None，dict 长度不变。"""
    r0, r1 = make_req(0), make_req(1)
    dm = make_manager(r0, r1)

    result = dm.abort_req(999)

    assert result is None, "should return None when uid not found"
    assert len(dm.running_reqs) == 2, "dict should be unchanged"


def test_abort_req_empty():
    """空 dict 时 abort_req 返回 None，不抛异常。"""
    dm = DecodeManager(page_size=4)
    assert dm.abort_req(0) is None


# ---------------------------------------------------------------------------
# remove_req
# ---------------------------------------------------------------------------

def test_remove_req_hit():
    """remove_req 正常移除存在的 Req。"""
    r0, r1 = make_req(0), make_req(1)
    dm = make_manager(r0, r1)

    dm.remove_req(r0)

    assert 0 not in dm.running_reqs
    assert len(dm.running_reqs) == 1


def test_remove_req_idempotent():
    """重复 remove 同一个 Req 不抛异常（幂等）。"""
    r0 = make_req(0)
    dm = make_manager(r0)

    dm.remove_req(r0)
    dm.remove_req(r0)  # second call must not raise


# ---------------------------------------------------------------------------
# filter_reqs
# ---------------------------------------------------------------------------

def test_filter_reqs_merges_new():
    """新传入的 Req 被合并到 running_reqs。"""
    r0 = make_req(0)
    dm = make_manager(r0)

    r1 = make_req(1)
    dm.filter_reqs([r1])

    assert 0 in dm.running_reqs
    assert 1 in dm.running_reqs


def test_filter_reqs_removes_exhausted():
    """can_decode=False 的 Req（remain=0）在 filter_reqs 后被移除。"""
    r0 = make_req(0, remain=5)   # can_decode = True
    r1 = make_req(1, remain=0)   # can_decode = False
    dm = make_manager(r0, r1)

    dm.filter_reqs([])  # no new reqs, just trigger the filter

    assert 0 in dm.running_reqs, "r0 should survive"
    assert 1 not in dm.running_reqs, "r1 (exhausted) should be removed"


def test_filter_reqs_new_overwrites_old():
    """相同 uid 时，新传入的 Req 覆盖旧的。"""
    r_old = make_req(uid=7, remain=5)
    dm = make_manager(r_old)

    r_new = make_req(uid=7, remain=3)
    dm.filter_reqs([r_new])

    assert dm.running_reqs[7] is r_new, "new req should overwrite old"


# ---------------------------------------------------------------------------
# inflight_tokens
# ---------------------------------------------------------------------------

def test_inflight_tokens():
    """inflight_tokens = sum(remain_len) + (page_size-1) * count。"""
    page_size = 4
    r0 = make_req(0, remain=10)  # remain_len = 10
    r1 = make_req(1, remain=6)   # remain_len = 6
    dm = make_manager(r0, r1)
    dm.page_size = page_size

    expected = (10 + 6) + (page_size - 1) * 2  # 16 + 6 = 22
    assert dm.inflight_tokens == expected


# ---------------------------------------------------------------------------
# schedule_next_batch
# ---------------------------------------------------------------------------

def test_schedule_next_batch_returns_all_reqs():
    """schedule_next_batch 包含所有 running_reqs。"""
    r0, r1, r2 = make_req(0), make_req(1), make_req(2)
    dm = make_manager(r0, r1, r2)

    batch = dm.schedule_next_batch()

    assert batch is not None
    assert batch.phase == "decode"
    assert set(r.uid for r in batch.reqs) == {0, 1, 2}


def test_schedule_next_batch_empty():
    """running_reqs 为空时返回 None。"""
    dm = DecodeManager(page_size=4)
    assert dm.schedule_next_batch() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
