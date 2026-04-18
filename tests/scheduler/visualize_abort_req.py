"""
可视化 DecodeManager 请求生命周期（Set → Dict 重构后）。

模拟 add → decode → abort → finish 全流程，
每步打印 running_reqs 的状态快照，展示 O(1) Dict 查找的工作方式。

运行方式：
    .venv/bin/python tests/scheduler/visualize_abort_req.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

from minisgl.core import Req, SamplingParams

# 直接加载 decode.py，绕开 scheduler/__init__.py（会拉入 zmq/msgpack 等重依赖）
_decode_path = Path(__file__).parents[2] / "python/minisgl/scheduler/decode.py"
_spec = importlib.util.spec_from_file_location("minisgl.scheduler.decode", _decode_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
sys.modules["minisgl.scheduler.decode"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore
DecodeManager = _mod.DecodeManager

# ------------------------------------------------------------------ #
# ANSI 颜色
# ------------------------------------------------------------------ #
RESET  = "\033[0m"
BOLD   = "\033[1m"
GRAY   = "\033[90m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"


# ------------------------------------------------------------------ #
# 宽字符对齐工具（中文字符显示宽度=2，英文=1）
# ------------------------------------------------------------------ #
import unicodedata

def display_width(s: str) -> int:
    """计算字符串的终端显示宽度（中文/全角=2，其他=1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def pad_center(s: str, width: int, fill: str = " ") -> str:
    """按显示宽度居中对齐。"""
    w = display_width(s)
    total_pad = max(0, width - w)
    left = total_pad // 2
    right = total_pad - left
    return fill * left + s + fill * right


def pad_right(s: str, width: int, fill: str = " ") -> str:
    """按显示宽度右对齐。"""
    w = display_width(s)
    return fill * max(0, width - w) + s


def section(title: str) -> None:
    width = 62
    print()
    print(BOLD + CYAN + "=" * width + RESET)
    print(BOLD + CYAN + f"  {title}" + RESET)
    print(BOLD + CYAN + "=" * width + RESET)


def step(n: int, desc: str) -> None:
    print()
    print(BOLD + YELLOW + f"Step {n}: {desc}" + RESET)
    print(YELLOW + "-" * 54 + RESET)


def print_dict_state(dm: DecodeManager) -> None:
    """打印 running_reqs 当前状态表格。纯 ASCII 分隔符，避免 box-drawing 字符宽度不一致。"""
    # 列宽（纯 ASCII，>N 精确可靠）
    # uid:4  remain_len:10  can_decode:10
    print(f"  running_reqs 状态:")
    print(f"  {'uid':>4}  |  {'remain_len':>10}  |  {'can_decode':>10}")
    print(f"  {'':>4}  |  {'(max left)':>10}  |  {'(>0: run)':>10}")
    print(f"  {'-'*4}--+--{'-'*10}--+--{'-'*10}")
    if not dm.running_reqs:
        print(GRAY + "  (空)" + RESET)
    for uid, req in dm.running_reqs.items():
        cd_str = GREEN + "True " + RESET if req.can_decode else RED + "False" + RESET
        print(f"  {uid:>4}  |  {req.remain_len:>10}  |  {cd_str}")
    count = len(dm.running_reqs)
    color = GREEN if count > 0 else GRAY
    print(f"  {color}(共 {count} 个请求，runnable={dm.runnable}){RESET}")


# ------------------------------------------------------------------ #
# 辅助
# ------------------------------------------------------------------ #
def make_req(uid: int, remain: int) -> Req:
    return Req(
        input_ids=torch.tensor([uid + 1], dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=remain,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=MagicMock(),
    )


def exhaust(req: Req) -> None:
    """把 req 消耗到 remain_len=0，模拟 decode 完成。"""
    req.device_len = req.max_device_len


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
def main() -> None:
    section("DecodeManager 请求生命周期可视化（Dict 重构后）")

    dm = DecodeManager(page_size=4)
    r0 = make_req(uid=0, remain=10)
    r1 = make_req(uid=1, remain=8)
    r2 = make_req(uid=2, remain=6)

    # ── 概念说明：remain_len ─────────────────────────────────────────
    section("概念：remain_len 是什么？")
    print(f"  每次 API 调用对应一个 {BOLD}Req{RESET}，Req 记录了这次请求的序列状态。")
    print(f"  （多轮对话的每一轮 = 一次 API 调用 = 一个 Req，")
    print(f"   每次都把完整对话历史打包进 input_ids）")
    print()
    print(f"  序列长度示意（以 r0 为例，prompt=1 token，max_tokens=10）：")
    print()
    print(f"  input_ids:   [ t0 | g1  g2  g3 ... g10 ]")
    print(f"                 ↑      ↑                ↑")
    print(f"              prompt  已生成           最多到这")
    print()
    print(f"  {BOLD}字段对照表:{RESET}")
    # 列宽（显示宽度）
    c1, c2, c3 = 16, 32, 10
    print(f"  {pad_center('字段', c1)}  {pad_center('含义', c2)}  {pad_center('r0 初始值', c3)}")
    print(f"  {'─'*c1}  {'─'*c2}  {'─'*c3}")
    rows = [
        ("device_len",    "当前序列在 GPU 上的长度",       str(r0.device_len),      False, False),
        ("max_device_len","= device_len初始 + max_tokens", str(r0.max_device_len),  False, False),
        ("remain_len",    "= max_device_len - device_len", str(r0.remain_len),      True,  True),
        ("can_decode",    "= remain_len > 0",              str(r0.can_decode),      False, False),
    ]
    for field, meaning, val, bold, green in rows:
        f_str = pad_center(field,   c1)
        m_str = pad_center(meaning, c2)
        v_str = pad_center(val,     c3)
        if bold:
            f_str = BOLD + f_str + RESET
            m_str = BOLD + m_str + RESET
        if green:
            v_str = GREEN + v_str + RESET
        print(f"  {f_str}  {m_str}  {v_str}")
    print()
    print(f"  {YELLOW}每生成一个 token：device_len +1，remain_len -1{RESET}")
    print(f"  {YELLOW}remain_len 归零 → can_decode=False → 请求完成{RESET}")
    print()
    print(f"  {GRAY}注意：remain_len 是上限。若模型提前输出 EOS token，{RESET}")
    print(f"  {GRAY}请求也会结束，实际生成数 ≤ remain_len。{RESET}")

    # ── Step 1 ──────────────────────────────────────────────────────
    step(1, "filter_reqs([r0, r1, r2]) — 添加 3 个请求进入 decode 阶段")
    print(f"  操作: for req in reqs: running_reqs[req.uid] = req")
    print(f"        {{ 0: r0, 1: r1, 2: r2 }}")
    dm.filter_reqs([r0, r1, r2])
    print()
    print_dict_state(dm)

    # ── Step 2 ──────────────────────────────────────────────────────
    step(2, "schedule_next_batch() — 调度生成 decode Batch")
    batch = dm.schedule_next_batch()
    print(f"  操作: Batch(reqs=list(running_reqs.values()), phase='decode')")
    print()
    if batch:
        uids = [r.uid for r in batch.reqs]
        print(f"  phase : {GREEN}{batch.phase}{RESET}")
        print(f"  reqs  : {GREEN}{uids}{RESET}  (共 {len(uids)} 个)")
        print(f"  inflight_tokens : {GREEN}{dm.inflight_tokens}{RESET}")
        print(GRAY + f"    = sum(remain_len) + (page_size-1)*count" + RESET)
        print(GRAY + f"    = {sum(r.remain_len for r in dm.running_reqs.values())}"
              f" + {dm.page_size - 1}×{len(dm.running_reqs)}"
              f" = {dm.inflight_tokens}" + RESET)

    # ── Step 3 ──────────────────────────────────────────────────────
    step(3, "abort_req(uid=1) — 用户主动取消请求 1")
    print(f"  操作: running_reqs.pop(1, None)  →  O(1) 直接命中")
    result = dm.abort_req(1)
    print()
    if result is not None:
        print(f"  返回: {GREEN}Req(uid={result.uid}, remain_len={result.remain_len}, "
              f"can_decode={result.can_decode}){RESET}  {GREEN}✓ 命中{RESET}")
    else:
        print(f"  返回: {RED}None  ✗ 未命中{RESET}")
    print()
    print_dict_state(dm)

    # ── Step 4 ──────────────────────────────────────────────────────
    step(4, "abort_req(uid=999) — 取消不存在的请求")
    print(f"  操作: running_reqs.pop(999, None)  →  key 不存在，返回 None")
    result = dm.abort_req(999)
    print()
    if result is None:
        print(f"  返回: {GRAY}None  (uid=999 不在 dict 中){RESET}  {GREEN}✓ 安全，不抛异常{RESET}")
    print()
    print_dict_state(dm)

    # ── Step 5 ──────────────────────────────────────────────────────
    step(5, "r0 解码完成（remain=0），filter_reqs([]) 触发过滤")
    print(f"  操作: exhaust(r0)  →  r0.remain_len = 0, r0.can_decode = False")
    exhaust(r0)
    print(f"  操作: filter_reqs([])  →  过滤掉 can_decode=False 的请求")
    dm.filter_reqs([])
    print()
    print(f"  r0: remain_len={r0.remain_len}, can_decode={RED}{r0.can_decode}{RESET}  {RED}✗ 被移除{RESET}")
    print(f"  r2: remain_len={r2.remain_len}, can_decode={GREEN}{r2.can_decode}{RESET}  {GREEN}✓ 保留{RESET}")
    print()
    print_dict_state(dm)

    # ── Step 6 ──────────────────────────────────────────────────────
    step(6, "remove_req(r2) — 请求 2 正常完成，移出 decode 队列")
    print(f"  操作: running_reqs.pop(r2.uid, None)  →  pop(2, None)")
    dm.remove_req(r2)
    print()
    print_dict_state(dm)

    # ── 幂等性演示 ───────────────────────────────────────────────────
    print()
    print(BOLD + YELLOW + "  额外验证: 对已移除的 r2 再次 remove_req — 不抛异常（幂等）" + RESET)
    dm.remove_req(r2)
    print(f"  {GREEN}✓ 安全{RESET}")

    # ── 总结 ─────────────────────────────────────────────────────────
    print()
    print(BOLD + CYAN + "=" * 62 + RESET)
    print(BOLD + CYAN + "  生命周期完成：add → decode → abort/finish → 队列清空" + RESET)
    print(BOLD + CYAN + "=" * 62 + RESET)
    print()


if __name__ == "__main__":
    main()
