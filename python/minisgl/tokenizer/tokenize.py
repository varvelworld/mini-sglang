from __future__ import annotations

import warnings
from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
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

        # batch tokenization
        try:
            # 批量Encode
            enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
            for i in range(len(prompts)) :
                mask = enc["attention_mask"][i].bool()
                inputs = enc["input_ids"][i][mask].to(torch.int32)
                results.append(inputs)
        except Exception as e:
            results = []
            warnings.warn(f"Batch tokenization failed with error: {e}. Falling back to individual tokenization.")
            # 逐条Fallback Encode
            for prompt in prompts:
                input_ids: torch.Tensor = (  # type: ignore
                    self.tokenizer.encode(prompt, return_tensors="pt")
                )
                results.append(input_ids.view(-1).to(torch.int32))
        return results
