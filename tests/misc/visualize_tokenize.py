"""
可视化 batch tokenization 的逐步数据变换过程。

运行方式：
    .venv/bin/python tests/misc/visualize_tokenize.py
"""
from __future__ import annotations

from transformers import AutoTokenizer

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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # gpt2 没有 pad_token，用 eos 代替

    # ------------------------------------------------------------------ #
    # 输入
    # ------------------------------------------------------------------ #
    section("Batch Tokenization 可视化")

    prompts = [
        "hello",
        "hello world, this is a longer sentence",
        "the quick brown fox",
    ]

    step(1, "原始输入文本")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {GREEN}\"{p}\"{RESET}  (字符数: {len(p)})")

    # ------------------------------------------------------------------ #
    # Step 2: 逐条 encode（对比基准）
    # ------------------------------------------------------------------ #
    step(2, "逐条 encode 的结果（基准对比）")
    single_results = []
    for i, p in enumerate(prompts):
        ids = tokenizer.encode(p)
        single_results.append(ids)
        tokens = [tokenizer.decode([t]) for t in ids]
        print(f"  [{i}] ids={ids}")
        print(f"       tokens={tokens}")

    # ------------------------------------------------------------------ #
    # Step 3: 批量 encode（带 padding）
    # ------------------------------------------------------------------ #
    step(3, "批量 encode（padding=True）— 原始输出")
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids   = enc["input_ids"]
    attn_mask   = enc["attention_mask"]
    pad_id      = tokenizer.pad_token_id
    max_len     = input_ids.shape[1]

    print(f"  batch shape: {list(input_ids.shape)}  (batch_size x max_len={max_len})")
    print(f"  pad_token_id: {pad_id} = \"{tokenizer.decode([pad_id])}\"")
    print()
    print(f"  {'':>4}  " + "  ".join(f"{j:>6}" for j in range(max_len)))
    print(f"  {'':>4}  " + "  ".join("-" * 6 for _ in range(max_len)))
    for i in range(len(prompts)):
        row = []
        for j in range(max_len):
            token_id = input_ids[i, j].item()
            is_pad   = attn_mask[i, j].item() == 0
            cell     = f"{token_id:>6}"
            row.append(GRAY + cell + RESET if is_pad else GREEN + cell + RESET)
        print(f"  [{i}]   " + "  ".join(row))

    # ------------------------------------------------------------------ #
    # Step 4: attention_mask
    # ------------------------------------------------------------------ #
    step(4, "attention_mask（1=真实 token，0=padding）")
    print(f"  {'':>4}  " + "  ".join(f"{j:>6}" for j in range(max_len)))
    print(f"  {'':>4}  " + "  ".join("-" * 6 for _ in range(max_len)))
    for i in range(len(prompts)):
        row = []
        for j in range(max_len):
            val    = attn_mask[i, j].item()
            cell   = f"{val:>6}"
            row.append(GREEN + cell + RESET if val == 1 else RED + cell + RESET)
        print(f"  [{i}]   " + "  ".join(row))

    # ------------------------------------------------------------------ #
    # Step 5: 用 mask 过滤，还原真实 token ids
    # ------------------------------------------------------------------ #
    step(5, "用 mask 过滤 padding，还原真实 token ids")
    batch_results = []
    for i in range(len(prompts)):
        mask   = attn_mask[i].bool()
        ids    = input_ids[i][mask].tolist()
        tokens = [tokenizer.decode([t]) for t in ids]
        batch_results.append(ids)
        print(f"  [{i}] ids={ids}")
        print(f"       tokens={tokens}")

    # ------------------------------------------------------------------ #
    # Step 6: 和逐条 encode 对比验证
    # ------------------------------------------------------------------ #
    step(6, "验证：批量结果 == 逐条 encode 结果")
    all_ok = True
    for i in range(len(prompts)):
        match = batch_results[i] == single_results[i]
        status = GREEN + "✓ match" + RESET if match else RED + "✗ mismatch" + RESET
        print(f"  [{i}] {status}")
        if not match:
            all_ok = False
            print(f"       batch:  {batch_results[i]}")
            print(f"       single: {single_results[i]}")

    print()
    if all_ok:
        print(BOLD + GREEN + "  全部一致，batch tokenization 正确！" + RESET)
    else:
        print(BOLD + RED + "  存在不一致，请检查！" + RESET)
    print()


if __name__ == "__main__":
    main()
