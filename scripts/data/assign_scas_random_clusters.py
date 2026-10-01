#!/usr/bin/env python3
"""Assign SCAS traces to uniform-random clusters for K=2..10 (baseline).

Writes assignments_random_kXX.parquet (+ val) under --output-dir.
Cluster IDs are i.i.d. Uniform{0..K-1}; style_1 naming (by mean length) is
applied later at dataset-build time, matching other cluster pipelines.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def answer_char_lens(df: pd.DataFrame) -> np.ndarray:
    if "answer" in df.columns and df["answer"].notna().all():
        return df["answer"].astype(str).str.len().to_numpy(dtype=np.int32)
    lens = np.empty(len(df), dtype=np.int32)
    for i, messages in enumerate(df["messages"].to_numpy()):
        if isinstance(messages, str):
            messages = json.loads(messages)
        lens[i] = len(messages[1]["content"])
    return lens


def assign_split(df: pd.DataFrame, k: int, rng: np.random.RandomState) -> pd.DataFrame:
    n = len(df)
    cluster_id = rng.randint(0, k, size=n).astype(np.int32)
    lens = answer_char_lens(df)
    return pd.DataFrame(
        {
            "source_dataset": df["source_dataset"].astype(str).to_numpy(),
            "question_id": df["question_id"].astype(str).to_numpy(),
            "style_name": df["style_name"].astype(str).to_numpy(),
            "style_id": df["style_id"].astype(np.int32).to_numpy(),
            "cluster_id": cluster_id,
            "answer_char_len": lens,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path("data/scas/modc_prefix_full_sft"),
        help="Source ModC-prefix SFT parquets (same row identity as AE cluster builds)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scas/cluster_random_assignments-qwen3-4b"),
    )
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_parquet(args.prefix_dir / "train.parquet")
    val = pd.read_parquet(args.prefix_dir / "validation.parquet")
    print(f"Loaded train={len(train)} val={len(val)} from {args.prefix_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = []
    for k in range(args.k_min, args.k_max + 1):
        # Independent RNG streams per K (and train vs val) for reproducibility
        rng_tr = np.random.RandomState(args.seed + 1000 * k)
        rng_va = np.random.RandomState(args.seed + 1000 * k + 1)
        tr_as = assign_split(train, k, rng_tr)
        va_as = assign_split(val, k, rng_va)
        tr_path = args.output_dir / f"assignments_random_k{k:02d}.parquet"
        va_path = args.output_dir / f"assignments_random_val_k{k:02d}.parquet"
        tr_as.to_parquet(tr_path, index=False)
        va_as.to_parquet(va_path, index=False)
        counts = dict(Counter(int(x) for x in tr_as["cluster_id"]))
        mean_len = {
            str(c): float(tr_as.loc[tr_as["cluster_id"] == c, "answer_char_len"].mean())
            for c in range(k)
        }
        row = {
            "k": k,
            "train_counts": {str(c): counts.get(c, 0) for c in range(k)},
            "val_counts": dict(Counter(int(x) for x in va_as["cluster_id"])),
            "mean_answer_char_len_by_cluster": mean_len,
            "train_path": str(tr_path),
            "val_path": str(va_path),
        }
        sweep.append(row)
        print(f"K={k}: train sizes={[counts.get(c, 0) for c in range(k)]}")

    meta = {
        "method": "uniform_random_cluster_assignment",
        "prefix_dir": str(args.prefix_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "note": (
            "Each row independently Uniform{0..K-1}. Not content-based. "
            "Baseline for prefix-conditioning experiments."
        ),
        "sweep": sweep,
    }
    meta_path = args.output_dir / "random_cluster_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
