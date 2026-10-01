#!/usr/bin/env python3
"""Build control / method-fix variants of a [style_i] prefix SFT dataset.

Variants
  const1   : every prefix rewritten to [style_1]   (zero partition information)
  balanced : resample rows so every style has equal count (fixes the 41% dominant cluster)
  special  : [style_i] -> <style_i>                (dedicated special-token channel)

Source and output follow the cluster_prefix layout: <dir>/{train,validation}.parquet
with a `messages` column that is either a JSON string or an array of dicts. The
original encoding is preserved on write so downstream tokenization is unchanged.
"""

from __future__ import annotations
import os

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))
from scripts.scas_style_prefix import (  # noqa: E402
    HARD_PREFIX_RE,
    hard_to_soft_special_user_content,
)


def decode_msgs(raw):
    """Return (list_of_dicts, was_json_string)."""
    if isinstance(raw, str):
        return json.loads(raw), True
    if hasattr(raw, "tolist"):
        return [dict(m) for m in raw.tolist()], False
    return [dict(m) for m in raw], False


def encode_msgs(msgs, was_str):
    return json.dumps(msgs) if was_str else np.array(msgs, dtype=object)


def user_idx(msgs) -> int:
    for i, m in enumerate(msgs):
        if m.get("role") == "user":
            return i
    raise ValueError("no user turn found")


def get_style(raw) -> str:
    msgs, _ = decode_msgs(raw)
    m = HARD_PREFIX_RE.match(msgs[user_idx(msgs)]["content"])
    if not m:
        raise ValueError("no hard [style] prefix on user turn")
    return m.group(1)


def rewrite(raw, fn):
    msgs, was_str = decode_msgs(raw)
    i = user_idx(msgs)
    msgs[i] = dict(msgs[i])
    msgs[i]["content"] = fn(msgs[i]["content"])
    return encode_msgs(msgs, was_str)


def to_const1(content: str) -> str:
    return "[style_1]\n" + HARD_PREFIX_RE.sub("", content, count=1)


def transform(df: pd.DataFrame, variant: str, seed: int) -> tuple[pd.DataFrame, dict]:
    info: dict = {"variant": variant, "rows_in": int(len(df))}

    if variant == "const1":
        df = df.copy()
        df["messages"] = [rewrite(m, to_const1) for m in df["messages"]]
        for col in ("style_name", "cluster_style_name"):
            if col in df.columns:
                df[col] = "style_1"
        if "style_id" in df.columns:
            df["style_id"] = 0

    elif variant == "special":
        df = df.copy()
        df["messages"] = [rewrite(m, hard_to_soft_special_user_content) for m in df["messages"]]

    elif variant == "balanced":
        styles = np.array([get_style(m) for m in df["messages"]])
        counts = Counter(styles)
        target = int(np.median(list(counts.values())))
        rng = np.random.default_rng(seed)
        df = df.copy()
        df["_style"] = styles
        parts = []
        for st, grp in df.groupby("_style", sort=True):
            idx = rng.choice(len(grp), size=target, replace=len(grp) < target)
            parts.append(grp.iloc[idx])
        df = (
            pd.concat(parts)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
            .drop(columns=["_style"])
        )
        info["target_per_style"] = target
        info["counts_in"] = {k: int(v) for k, v in sorted(counts.items())}

    else:
        raise ValueError(f"unknown variant {variant}")

    info["rows_out"] = int(len(df))
    if variant != "special":
        out_counts = Counter(get_style(m) for m in df["messages"])
        info["counts_out"] = {k: int(v) for k, v in sorted(out_counts.items())}
    return df, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True, choices=["const1", "balanced", "special"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"src": str(src), "variant": args.variant, "seed": args.seed, "splits": {}}

    for split in ("train", "validation"):
        f = src / f"{split}.parquet"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        df = pd.read_parquet(f)
        # Resampling the validation split would change the eval set, so only the
        # prefix-format rewrites are applied there.
        if split == "validation" and args.variant == "balanced":
            newdf, info = df, {"variant": "passthrough", "rows_in": len(df), "rows_out": len(df)}
        else:
            newdf, info = transform(df, args.variant, args.seed)
        newdf.to_parquet(out / f"{split}.parquet", index=False)
        meta["splits"][split] = info
        print(f"{split}: {json.dumps(info)}", flush=True)

    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
