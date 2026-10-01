"""Pass@k helpers for K-way discrete prefix conditioning."""

from __future__ import annotations

import json
from pathlib import Path


def style_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("style_") and name.split("_", 1)[1].isdigit():
        return (int(name.split("_", 1)[1]), name)
    return (10**9, name)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    styles = sorted(weights.keys(), key=style_sort_key)
    total = float(sum(weights[s] for s in styles))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {s: float(weights[s] / total) for s in styles}


def allocate_integer_budget(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of `total` samples across styles."""
    if total <= 0:
        raise ValueError("total must be positive")
    w = normalize_weights(weights)
    styles = sorted(w.keys(), key=style_sort_key)
    raw = {s: total * w[s] for s in styles}
    counts = {s: int(raw[s]) for s in styles}
    remainder = total - sum(counts.values())
    if remainder:
        order = sorted(styles, key=lambda s: (raw[s] - counts[s]), reverse=True)
        for s in order[:remainder]:
            counts[s] += 1
    if total >= len(styles):
        for s in styles:
            if counts[s] == 0:
                donor = max(styles, key=lambda x: counts[x])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[s] = 1
    return counts


def multi_prefix_pass_at_k(p_by_style: dict[str, float], k: int, weights: dict[str, float]) -> float:
    """K-prefix Pass@k with budget split proportional to `weights`."""
    if k <= 0:
        return 0.0
    w = normalize_weights(weights)
    styles = sorted(w.keys(), key=style_sort_key)
    if k == 1:
        return float(sum(w[s] * float(p_by_style.get(s, 0.0)) for s in styles))
    alloc = allocate_integer_budget(k, w)
    fail = 1.0
    for s in styles:
        p = float(p_by_style.get(s, 0.0))
        fail *= (1.0 - p) ** alloc[s]
    return float(1.0 - fail)


def balanced_weights(style_names: list[str]) -> dict[str, float]:
    n = len(style_names)
    if n == 0:
        raise ValueError("style_names must be non-empty")
    return {s: 1.0 / n for s in style_names}


def load_cluster_meta(meta_path: Path) -> dict:
    data = json.loads(meta_path.read_text())
    style_names = data.get("style_names")
    if not style_names:
        style_names = sorted(
            data.get("train_summary", {}).get("style_counts", {}).keys(),
            key=style_sort_key,
        )
    if not style_names:
        raise ValueError(f"No style_names in {meta_path}")
    train_counts = data.get("train_summary", {}).get("style_counts", {})
    if not train_counts:
        raise ValueError(f"Missing train_summary.style_counts in {meta_path}")
    total = float(sum(train_counts[s] for s in style_names))
    train_weights = {s: float(train_counts[s] / total) for s in style_names}
    return {
        "style_names": style_names,
        "train_weights": train_weights,
        "n_clusters": int(data.get("n_clusters", len(style_names))),
        "raw": data,
    }
