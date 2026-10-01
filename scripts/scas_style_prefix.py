"""Shared style-prefix formatting for hard [style_i] vs soft-special <style_i>."""

from __future__ import annotations

import re

HARD_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\n")
SOFT_SPECIAL_PREFIX_RE = re.compile(r"^<(style_\d+)>\n")
ANY_STYLE_PREFIX_RE = re.compile(r"^(?:\[[^\]]+\]|<(style_\d+)>)\n")


def style_token_hard(style_name: str) -> str:
    """Literal multi-subword cue used in current SCAS cluster trains."""
    return f"[{style_name}]"


def style_token_soft_special(style_name: str) -> str:
    """Single dedicated special-token cue, e.g. <style_1>."""
    # style_name is already "style_i"
    return f"<{style_name}>"


def format_style_prefix(style_name: str, prefix_mode: str) -> str:
    if prefix_mode in {"hard", "bracket"}:
        return style_token_hard(style_name)
    if prefix_mode in {"soft_special", "soft-special", "special"}:
        return style_token_soft_special(style_name)
    raise ValueError(f"Unknown prefix_mode={prefix_mode!r} (expected hard|soft_special)")


def user_content_with_style(style_name: str, question: str, prefix_mode: str) -> str:
    return f"{format_style_prefix(style_name, prefix_mode)}\n{question}"


def strip_style_prefix(user_content: str) -> str:
    """Remove a leading hard or soft-special style prefix from user content."""
    return ANY_STYLE_PREFIX_RE.sub("", user_content, count=1)


def soft_special_token_list(k: int) -> list[str]:
    return [f"<style_{i}>" for i in range(1, int(k) + 1)]


def hard_to_soft_special_user_content(user_content: str) -> str:
    """Rewrite leading [style_i]\\n → <style_i>\\n."""
    m = HARD_PREFIX_RE.match(user_content)
    if not m:
        raise ValueError(f"Expected hard [style] prefix, got: {user_content[:80]!r}")
    name = m.group(1)
    if not re.fullmatch(r"style_\d+", name):
        raise ValueError(f"Expected style_N name inside brackets, got {name!r}")
    return f"<{name}>\n{user_content[m.end():]}"
