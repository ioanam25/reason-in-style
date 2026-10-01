"""Shared tokenizer helpers for SCAS data / SFT builders."""

from __future__ import annotations


def configure_tokenizer(model_name: str, tokenizer):
    """Pad token + chat template fallback (Qwen / Llama / generic)."""
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    name = (model_name or "").lower()
    if "qwen" in name:
        tokenizer.chat_template = (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
        )
    else:
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
