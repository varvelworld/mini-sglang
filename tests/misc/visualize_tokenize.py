"""
可视化 batch tokenization 的逐步数据变换过程。
通过 PrintHook 回调展示 TokenizeManager 内部每一步的真实数据。

运行方式：
    .venv/bin/python tests/misc/visualize_tokenize.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

# 直接加载 tokenize.py，避免触发 tokenizer/__init__.py（会拉入 ZMQ 等重依赖）
_tokenize_path = Path(__file__).parents[2] / "python/minisgl/tokenizer/tokenize.py"
_spec = importlib.util.spec_from_file_location("minisgl.tokenizer.tokenize", _tokenize_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
sys.modules["minisgl.tokenizer.tokenize"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore
TokenizeManager = _mod.TokenizeManager

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg

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
    width = 60
    print()
    print(BOLD + CYAN + "=" * width + RESET)
    print(BOLD + CYAN + f"  {title}" + RESET)
    print(BOLD + CYAN + "=" * width + RESET)


def step(n: int, desc: str) -> None:
    print()
    print(BOLD + YELLOW + f"Step {n}: {desc}" + RESET)
    print(YELLOW + "-" * 50 + RESET)


# ------------------------------------------------------------------ #
# Hook 实现
# ------------------------------------------------------------------ #
class PrintHook:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._step = 1

    def on_prompts(self, prompts: List[str]) -> None:
        step(self._step, "原始输入文本（chat template 展开后）")
        self._step += 1
        for i, p in enumerate(prompts):
            print(f"  [{i}] {GREEN}\"{p}\"{RESET}  (字符数: {len(p)})")

    def on_batch_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        step(self._step, "批量 encode 原始输出（含 padding）")
        self._step += 1
        pad_id  = self.tokenizer.pad_token_id
        max_len = input_ids.shape[1]
        n       = input_ids.shape[0]

        print(f"  shape: {list(input_ids.shape)}  (batch_size x max_len={max_len})")
        print(f"  pad_token_id: {pad_id} = \"{self.tokenizer.decode([pad_id])}\"  (灰色表示 padding)")
        print()

        # input_ids 表格
        print(f"  input_ids:")
        print(f"  {'':>4}  " + "  ".join(f"{j:>6}" for j in range(max_len)))
        print(f"  {'':>4}  " + "  ".join("-" * 6 for _ in range(max_len)))
        for i in range(n):
            row = []
            for j in range(max_len):
                token_id = input_ids[i, j].item()
                is_pad   = attention_mask[i, j].item() == 0
                cell     = f"{token_id:>6}"
                row.append(GRAY + cell + RESET if is_pad else GREEN + cell + RESET)
            print(f"  [{i}]   " + "  ".join(row))

        # attention_mask 表格
        print()
        print(f"  attention_mask:  (绿=真实 token，红=padding)")
        print(f"  {'':>4}  " + "  ".join(f"{j:>6}" for j in range(max_len)))
        print(f"  {'':>4}  " + "  ".join("-" * 6 for _ in range(max_len)))
        for i in range(n):
            row = []
            for j in range(max_len):
                val  = attention_mask[i, j].item()
                cell = f"{val:>6}"
                row.append(GREEN + cell + RESET if val == 1 else RED + cell + RESET)
            print(f"  [{i}]   " + "  ".join(row))

    def on_results(self, results: List[torch.Tensor]) -> None:
        step(self._step, "用 attention_mask 过滤 padding → 最终结果")
        self._step += 1
        for i, ids in enumerate(results):
            tokens = [self.tokenizer.decode([t]) for t in ids.tolist()]
            print(f"  [{i}] ids={ids.tolist()}")
            print(f"       tokens={tokens}")

    def on_fallback(self, error: Exception) -> None:
        print()
        print(BOLD + RED + f"  [fallback] 批量编码失败，退回逐条 encode" + RESET)
        print(GRAY + f"  原因: {error}" + RESET)


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # gpt2 没有 pad_token，用 eos 代替

    msgs = [
        TokenizeMsg(uid=0, text="hello", sampling_params=SamplingParams()),
        TokenizeMsg(uid=1, text="hello world, this is a longer sentence", sampling_params=SamplingParams()),
        TokenizeMsg(uid=2, text="the quick brown fox", sampling_params=SamplingParams()),
    ]

    section("Batch Tokenization 可视化（via TokenizeManager + PrintHook）")

    hook = PrintHook(tokenizer)
    manager = TokenizeManager(tokenizer)
    results = manager.tokenize(msgs, hook=hook)

    # 最终验证
    step(hook._step, "验证：批量结果 == 逐条 encode 结果")
    all_ok = True
    for i, msg in enumerate(msgs):
        assert isinstance(msg.text, str)
        expected = tokenizer.encode(msg.text, return_tensors="pt")
        expected = expected.view(-1).to(torch.int32)
        match = torch.equal(results[i], expected)
        status = GREEN + "✓ match" + RESET if match else RED + "✗ mismatch" + RESET
        print(f"  [{i}] {status}")
        if not match:
            all_ok = False

    print()
    if all_ok:
        print(BOLD + GREEN + "  全部一致，batch tokenization 正确！" + RESET)
    else:
        print(BOLD + RED + "  存在不一致，请检查！" + RESET)
    print()


if __name__ == "__main__":
    main()
