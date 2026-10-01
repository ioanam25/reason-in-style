#!/usr/bin/env python3
"""Build [style_i] prefix SFT data from random-cluster labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PREFIX_RE = re.compile(r"^\[[^\]]+\]\n")


def cluster_id_to_style_name(cluster_id: int) -> str:
    return f"style_{int(cluster_id) + 1}"


def order_clusters_by_mean_len(cluster_ids: np.ndarray, lens: np.ndarray) -> dict[int, int]:
    buckets: dict[int, list[int]] = {}
    for cid, tl in zip(cluster_ids, lens):
        buckets.setdefault(int(cid), []).append(int(tl))
    # empty clusters (unlikely) get +inf so they rank last
    ranked = sorted(
        ((cid, float(np.mean(vals)) if vals else float("inf")) for cid, vals in buckets.items()),
        key=lambda kv: kv[1],
    )
    return {int(cid): rank for rank, (cid, _) in enumerate(ranked)}


def build_split(
    prefix_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    rank_map: dict[int, int],
) -> tuple[pd.DataFrame, dict]:
    assign = assign_df.rename(columns={"style_name": "oracle_style_name"})
    merged = prefix_df.merge(
        assign,
        left_on=["source_dataset", "question_id", "style_name"],
        right_on=["source_dataset", "question_id", "oracle_style_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(prefix_df):
        raise ValueError(f"join dropped rows: {len(merged)} vs {len(prefix_df)}")

    style_names: list[str] = []
    style_ids: list[int] = []
    new_messages: list[list[dict]] = []
    for _, row in merged.iterrows():
        style_idx = rank_map[int(row["cluster_id"])]
        style_name = cluster_id_to_style_name(style_idx)
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        question = PREFIX_RE.sub("", messages[0]["content"], count=1)
        new_messages.append(
            [
                {"role": "user", "content": f"[{style_name}]\n{question}"},
                {"role": "assistant", "content": messages[1]["content"]},
            ]
        )
        style_names.append(style_name)
        style_ids.append(style_idx)

    cols = {
        "messages": new_messages,
        "question_id": merged["question_id"].astype(str).to_numpy(),
        "style_id": style_ids,
        "style_name": style_names,
        "source_dataset": merged["source_dataset"].astype(str).to_numpy(),
        "cluster_id": merged["cluster_id"].to_numpy(),
        "cluster_style_name": style_names,
        "oracle_style_name": merged["oracle_style_name"].astype(str).to_numpy(),
    }
    if "answer" in merged.columns:
        cols["answer"] = merged["answer"].to_numpy()
    out = pd.DataFrame(cols)
    stats = {
        "rows": int(len(out)),
        "cluster_style_counts": dict(Counter(style_names)),
        "raw_cluster_counts": dict(Counter(int(x) for x in merged["cluster_id"])),
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--prefix-dir", type=Path, default=Path("data/scas/modc_prefix_full_sft"))
    p.add_argument("--random-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    train_assign = pd.read_parquet(args.random_dir / f"assignments_random_k{args.k:02d}.parquet")
    val_assign = pd.read_parquet(args.random_dir / f"assignments_random_val_k{args.k:02d}.parquet")
    for df in (train_assign, val_assign):
        df["question_id"] = df["question_id"].astype(str)
        df["source_dataset"] = df["source_dataset"].astype(str)
        df["style_name"] = df["style_name"].astype(str)

    prefix_train = pd.read_parquet(args.prefix_dir / "train.parquet")
    prefix_val = pd.read_parquet(args.prefix_dir / "validation.parquet")
    for df in (prefix_train, prefix_val):
        df["question_id"] = df["question_id"].astype(str)
        df["source_dataset"] = df["source_dataset"].astype(str)
        df["style_name"] = df["style_name"].astype(str)

    rank_map = order_clusters_by_mean_len(
        train_assign["cluster_id"].to_numpy(),
        train_assign["answer_char_len"].to_numpy(),
    )
    # ensure all k ids present in map
    for c in range(args.k):
        rank_map.setdefault(c, len(rank_map))

    train_out, train_stats = build_split(prefix_train, train_assign, rank_map)
    val_out, val_stats = build_split(prefix_val, val_assign, rank_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(args.output_dir / "train.parquet", index=False)
    val_out.to_parquet(args.output_dir / "validation.parquet", index=False)
    payload = {
        "k": args.k,
        "prompt_format": (
            f"[style_i] from UNIFORM RANDOM clusters K={args.k} "
            "(style_1=shortest mean answer_char_len among random buckets)"
        ),
        "random_dir": str(args.random_dir),
        "prefix_dir": str(args.prefix_dir),
        "cluster_rank_map_raw_to_style_index": {str(k): int(v) for k, v in rank_map.items()},
        "train": train_stats,
        "validation": val_stats,
        "baseline": "random_cluster_prefix",
    }
    (args.output_dir / "dataset_meta.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
