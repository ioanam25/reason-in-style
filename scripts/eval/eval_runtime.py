"""Shared Pass@k helpers used by the SCAS eval scripts."""

from __future__ import annotations

import json
from pathlib import Path


def standard_pass_at_k(p_pool: float, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(1.0 - (1.0 - p_pool) ** k)


def configure_tokenizer(tokenizer):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    tokenizer.chat_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{{bos_token}}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<|user|>\n{{message['content']}}\n"
        "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{message['content']}}{{eos_token}}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    return tokenizer


def load_model_config(checkpoint: str) -> dict:
    config_path = Path(checkpoint) / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text())
    return {}


def load_num_attention_heads(checkpoint: str) -> int:
    return int(load_model_config(checkpoint).get("num_attention_heads", 1))


def load_vocab_size(checkpoint: str) -> int | None:
    vocab_size = load_model_config(checkpoint).get("vocab_size")
    return int(vocab_size) if vocab_size is not None else None


def choose_tensor_parallel_size(
    n_gpus: int,
    num_attention_heads: int,
    requested: int,
    vocab_size: int | None = None,
) -> int:
    """Pick the largest TP size <= n_gpus that divides attention heads and vocab."""
    if n_gpus <= 0:
        return 1

    def tp_ok(tp: int) -> bool:
        if tp <= 0 or tp > n_gpus:
            return False
        if num_attention_heads % tp != 0:
            return False
        if vocab_size is not None and vocab_size % tp != 0:
            return False
        return True

    if requested > 0:
        if tp_ok(requested):
            return requested
        raise ValueError(
            f"tensor_parallel_size={requested} is incompatible with "
            f"{num_attention_heads} attention heads"
            + (f" and vocab_size={vocab_size}" if vocab_size is not None else "")
            + f" on {n_gpus} GPUs"
        )
    for tp in range(min(n_gpus, num_attention_heads), 0, -1):
        if tp_ok(tp):
            return tp
    return 1
