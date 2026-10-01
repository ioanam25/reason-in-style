#!/usr/bin/env python3
"""Copy gmm-filter-is-{rebal,obs,global} parquets and prepend [style_i] to user text.

Does not modify the original IS dirs. Val is also prefixed so valbest eval_loss
matches the train format (original IS val was prefix-stripped).
"""

from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.scas_style_prefix import strip_style_prefix, user_content_with_style  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
ARMS = ("rebal", "obs", "global")


def _msgs(raw) -> list[dict]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [dict(x) for x in raw]


def prefix_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_msgs = []
    for msgs, style in zip(out["messages"], out["style_name"]):
        m = _msgs(msgs)
        if not m or m[0].get("role") != "user":
            raise ValueError("expected messages[0].role == user")
        q = strip_style_prefix(str(m[0]["content"]))
        m[0]["content"] = user_content_with_style(str(style), q, "hard")
        new_msgs.append(m)
    out["messages"] = new_msgs
    return out


def main() -> None:
    for arm in ARMS:
        src = SROOT / "cluster_prefix" / f"gmm-filter-is-{arm}" / "k6"
        dst = SROOT / "cluster_prefix" / f"gmm-filter-is-{arm}-pfx" / "k6"
        if not (src / "train.parquet").is_file():
            raise FileNotFoundError(src / "train.parquet")
        dst.mkdir(parents=True, exist_ok=True)
        stats = {}
        for split in ("train", "validation"):
            df = pd.read_parquet(src / f"{split}.parquet")
            n = len(df)
            pref = prefix_split(df)
            # sanity: every user turn starts with [style_
            sample = pref.iloc[0]["messages"][0]["content"]
            if not sample.startswith("[style_"):
                raise RuntimeError(f"{arm} {split} missing prefix: {sample[:80]!r}")
            already = int(
                df["messages"].map(lambda m: str(_msgs(m)[0]["content"]).startswith("[style_")).sum()
            )
            pref.to_parquet(dst / f"{split}.parquet", index=False)
            stats[split] = {
                "n": n,
                "src_already_prefixed": already,
                "sample_user_prefix": sample.split("\n", 1)[0],
            }
            print(f"{arm:8s} {split:12s} n={n} src_prefixed={already} cue={stats[split]['sample_user_prefix']}")
        meta = {
            "src": str(src),
            "dst": str(dst),
            "variant": f"is-{arm}-pfx",
            "arm": f"gmm-filter-is-{arm}-pfx",
            "prefix_mode": "hard",
            "prompt_format": "[style_i]\\n + original IS question (weights unchanged)",
            "copied_weights": True,
            "splits": stats,
        }
        src_meta = src / "build_meta.json"
        if src_meta.is_file():
            meta["src_build_meta"] = json.loads(src_meta.read_text())
        (dst / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("wrote", dst)


if __name__ == "__main__":
    main()
