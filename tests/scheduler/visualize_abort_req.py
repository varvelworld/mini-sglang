"""
可视化 DecodeManager 请求生命周期（Set → Dict 重构后）。

通过 PrintHook 回调展示每个关键操作的内部状态变化，
与 visualize_tokenize.py 的 hook 模式保持一致。

运行方式：
    .venv/bin/python tests/scheduler/visualize_abort_req.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import torch
from minisgl.core import Batch, Req, SamplingParams

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


def _print_running(running: Dict[int, Req]) -> None:
    """打印 running_reqs 当前状态表格（纯 ASCII，对齐稳定）。"""
    print(f"  {'uid':>4}  |  {'remain_len':>10}  |  {'can_decode':>10}")
    print(f"  {'':>4}  |  {'(max left)':>10}  |  {'(>0: run)':>10}")
    print(f"  {'-'*4}--+--{'-'*10}--+--{'-'*10}")
    if not running:
        print(GRAY + "  (空)" + RESET)
    for uid, req in running.items():
        cd = GREEN + "True " + RESET if req.can_decode else RED + "False" + RESET
        print(f"  {uid:>4}  |  {req.remain_len:>10}  |  {cd}")
    count = len(running)
    color = GREEN if count > 0 else GRAY
    runnable = count > 0
    print(f"  {color}(共 {count} 个请求，runnable={runnable}){RESET}")


# ------------------------------------------------------------------ #
# PrintHook：实现 DecodeHook Protocol，负责所有打印
# ------------------------------------------------------------------ #
class PrintHook:
    def on_filter(
        self,
        added_uids: list[int],
        removed_uids: list[int],
        running: Dict[int, Req],
    ) -> None:
        if added_uids:
            print(f"  新加入: {GREEN}{added_uids}{RESET}")
        if removed_uids:
            print(f"  被过滤: {RED}{removed_uids}{RESET}  (can_decode=False)")
        else:
            print(f"  被过滤: {GRAY}无{RESET}")
        print()
        _print_running(running)

    def on_abort(self, uid: int, req: Req | None) -> None:
        if req is not None:
            print(f"  {GREEN}✓ 命中{RESET}  uid={uid} → "
                  f"Req(remain_len={req.remain_len}, can_decode={req.can_decode})")
        else:
            print(f"  {GRAY}✗ 未命中{RESET}  uid={uid} 不在 running_reqs 中，返回 None")

    def on_remove(self, req: Req) -> None:
        print(f"  移除 uid={req.uid}  (remain_len={req.remain_len})")

    def on_schedule(self, batch: Batch | None) -> None:
        if batch is None:
            print(f"  {GRAY}running_reqs 为空，返回 None{RESET}")
            return
        uids = [r.uid for r in batch.reqs]
        total_remain = sum(r.remain_len for r in batch.reqs)
        print(f"  phase  : {GREEN}{batch.phase}{RESET}")
        print(f"  reqs   : {GREEN}{uids}{RESET}  (共 {len(uids)} 个)")
        print(f"  sum(remain_len) = {total_remain}")


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

    # ── 概念说明：remain_len ─────────────────────────────────────────
    section("概念：remain_len 是什么？")
    r0_demo = make_req(uid=0, remain=10)
    print(f"  每次 API 调用对应一个 {BOLD}Req{RESET}，Req 记录了这次请求的序列状态。")
    print("  （多轮对话的每一轮 = 一次 API 调用 = 一个 Req，")
    print("   每次都把完整对话历史打包进 input_ids）")
    print()
    print("  序列长度示意（以 r0 为例，prompt=1 token，max_tokens=10）：")
    print()
    print("  input_ids:   [ t0 | g1  g2  g3 ... g10 ]")
    print("                 |      |                |")
    print("              prompt  已生成           最多到这")
    print()
    print(f"  {BOLD}字段对照表:{RESET}")
    c1, c2, c3 = 16, 34, 10
    print(f"  {'字段':^{c1}}  {'含义':^{c2}}  {'r0 初始值':^{c3}}")
    print(f"  {'-'*c1}  {'-'*c2}  {'-'*c3}")
    rows = [
        ("device_len",     "当前序列长度（prompt + 已生成）",  str(r0_demo.device_len),     False, False),
        ("max_device_len", "= device_len_init + max_tokens",  str(r0_demo.max_device_len), False, False),
        ("remain_len",     "= max_device_len - device_len",   str(r0_demo.remain_len),     True,  True),
        ("can_decode",     "= remain_len > 0",                str(r0_demo.can_decode),     False, False),
    ]
    for fname, meaning, val, bold, green in rows:
        f_s = f"{fname:^{c1}}"
        m_s = f"{meaning:^{c2}}"
        v_s = f"{val:^{c3}}"
        if bold:
            f_s = BOLD + f_s + RESET
            m_s = BOLD + m_s + RESET
        if green:
            v_s = GREEN + v_s + RESET
        print(f"  {f_s}  {m_s}  {v_s}")
    print()
    print(f"  {YELLOW}每生成一个 token：device_len +1，remain_len -1{RESET}")
    print(f"  {YELLOW}remain_len 归零 → can_decode=False → 请求完成{RESET}")
    print()
    print(f"  {GRAY}注意：remain_len 是上限。若模型提前输出 EOS token，{RESET}")
    print(f"  {GRAY}请求也会结束，实际生成数 <= remain_len。{RESET}")

    # ── 初始化 ───────────────────────────────────────────────────────
    dm   = DecodeManager(page_size=4)
    hook = PrintHook()
    r0   = make_req(uid=0, remain=10)
    r1   = make_req(uid=1, remain=8)
    r2   = make_req(uid=2, remain=6)

    # ── Step 1 ──────────────────────────────────────────────────────
    step(1, "filter_reqs([r0, r1, r2]) — 添加 3 个请求进入 decode 阶段")
    print("  操作: running_reqs[req.uid] = req  (对每个传入的 req)")
    print()
    dm.filter_reqs([r0, r1, r2], hook=hook)

    # ── Step 2 ──────────────────────────────────────────────────────
    step(2, "schedule_next_batch() — 调度生成 decode Batch")
    print("  操作: Batch(reqs=list(running_reqs.values()), phase='decode')")
    print()
    dm.schedule_next_batch(hook=hook)
    print()
    print(GRAY + "  inflight_tokens = sum(remain_len) + (page_size-1)*count" + RESET)
    print(GRAY + f"                  = {sum(r.remain_len for r in dm.running_reqs.values())}"
          f" + {dm.page_size-1}x{len(dm.running_reqs)}"
          f" = {dm.inflight_tokens}" + RESET)

    # ── Step 3 ──────────────────────────────────────────────────────
    step(3, "abort_req(uid=1) — 用户主动取消请求 1")
    print("  操作: running_reqs.pop(1, None)  →  O(1) 直接命中")
    print()
    dm.abort_req(1, hook=hook)
    print()
    _print_running(dm.running_reqs)

    # ── Step 4 ──────────────────────────────────────────────────────
    step(4, "abort_req(uid=999) — 取消不存在的请求")
    print("  操作: running_reqs.pop(999, None)  →  key 不存在")
    print()
    dm.abort_req(999, hook=hook)

    # ── Step 5 ──────────────────────────────────────────────────────
    step(5, "r0 解码完成（remain=0），filter_reqs([]) 触发过滤")
    print("  操作: exhaust(r0) → r0.remain_len=0, r0.can_decode=False")
    print("        filter_reqs([]) → 过滤掉 can_decode=False 的请求")
    print()
    exhaust(r0)
    dm.filter_reqs([], hook=hook)

    # ── Step 6 ──────────────────────────────────────────────────────
    step(6, "remove_req(r2) — 请求 2 正常完成，移出 decode 队列")
    print("  操作: running_reqs.pop(r2.uid, None)")
    print()
    dm.remove_req(r2, hook=hook)
    print()
    _print_running(dm.running_reqs)

    # ── 幂等性 ───────────────────────────────────────────────────────
    print()
    print(BOLD + YELLOW + "  额外验证: 对已移除的 r2 再次 remove_req — 不抛异常（幂等）" + RESET)
    dm.remove_req(r2, hook=hook)
    print(f"  {GREEN}✓ 安全{RESET}")

    # ── 总结 ─────────────────────────────────────────────────────────
    print()
    print(BOLD + CYAN + "=" * 62 + RESET)
    print(BOLD + CYAN + "  生命周期完成：add → decode → abort/finish → 队列清空" + RESET)
    print(BOLD + CYAN + "=" * 62 + RESET)
    print()


if __name__ == "__main__":
    main()
