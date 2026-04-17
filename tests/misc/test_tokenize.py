from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

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

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_batch_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    msgs = [
        TokenizeMsg(uid=0, text="hello", sampling_params=SamplingParams()),
        TokenizeMsg(
            uid=1, text="hello world, this is a longer sentence", sampling_params=SamplingParams()
        ),
    ]

    manager = TokenizeManager(tokenizer)
    results = manager.tokenize(msgs)

    assert len(results) == len(msgs), "Output length mismatch"

    for i, msg in enumerate(msgs):
        assert isinstance(msg.text, str)
        expected = tokenizer.encode(msg.text, return_tensors="pt")
        expected = expected.view(-1).to(torch.int32)
        assert torch.equal(results[i], expected), (
            f"Mismatch at index {i}: {results[i]} vs {expected}"
        )

    logger.info("test_batch_tokenization passed")


def test_fallback_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = None  # 模拟没有 pad_token 的 tokenizer

    msgs = [
        TokenizeMsg(uid=0, text="hello", sampling_params=SamplingParams()),
        TokenizeMsg(
            uid=1, text="hello world, this is a longer sentence", sampling_params=SamplingParams()
        ),
    ]

    manager = TokenizeManager(tokenizer)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = manager.tokenize(msgs)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"

    assert len(results) == len(msgs), "Output length mismatch"

    for i, msg in enumerate(msgs):
        assert isinstance(msg.text, str)
        expected = tokenizer.encode(msg.text, return_tensors="pt")
        expected = expected.view(-1).to(torch.int32)
        assert torch.equal(results[i], expected), (
            f"Mismatch at index {i}: {results[i]} vs {expected}"
        )

    logger.info("test_fallback_tokenization passed")


if __name__ == "__main__":
    test_batch_tokenization()
    test_fallback_tokenization()
