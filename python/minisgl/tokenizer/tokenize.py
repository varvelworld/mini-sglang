from __future__ import annotations

import warnings
from typing import List, Protocol, runtime_checkable

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


@runtime_checkable
class TokenizeHook(Protocol):
    def on_prompts(self, prompts: List[str]) -> None:
        """Step 1 完成后：chat template 展开的 prompts 列表。"""
        ...

    def on_batch_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """Step 2 完成后：批量编码的原始输出（含 padding）。"""
        ...

    def on_results(self, results: List[torch.Tensor]) -> None:
        """Step 3 完成后：过滤 padding 后的最终结果。"""
        ...

    def on_fallback(self, error: Exception) -> None:
        """进入 fallback 路径时：触发异常的错误信息。"""
        ...


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(
        self,
        msgs: List[TokenizeMsg],
        hook: TokenizeHook | None = None,
    ) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []

        # Step 1: 展开 chat template → prompts
        prompts: List[str] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            prompts.append(prompt)

        if hook is not None:
            hook.on_prompts(prompts)

        # Step 2+3: 批量 encode，用 attention_mask 裁掉 padding
        try:
            enc = self.tokenizer(prompts, padding=True, return_tensors="pt")

            if hook is not None:
                hook.on_batch_encoded(enc["input_ids"], enc["attention_mask"])

            for i in range(len(prompts)):
                mask = enc["attention_mask"][i].bool()
                inputs = enc["input_ids"][i][mask].to(torch.int32)
                results.append(inputs)

        except Exception as e:
            results = []
            warnings.warn(
                f"Batch tokenization failed with error: {e}. Falling back to individual tokenization."
            )
            if hook is not None:
                hook.on_fallback(e)

            for prompt in prompts:
                input_ids: torch.Tensor = (  # type: ignore
                    self.tokenizer.encode(prompt, return_tensors="pt")
                )
                results.append(input_ids.view(-1).to(torch.int32))

        if hook is not None:
            hook.on_results(results)

        return results
