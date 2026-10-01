#!/usr/bin/env python3
"""Build [style_i] prefix SFT data from GMM cluster assignments (style_1=shortest)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.build_scas_kmeans_cluster_prefix_sft_dataset import (  # noqa: E402
    build_split,
    cluster_id_to_style_name,
    order_clusters_by_mean_len,
)


def load_train_assignments(gmm_dir: Path, k: int) -> pd.DataFrame:
    path = gmm_dir / f"assignments_gmm_k{k:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    parts = df["question_id"].astype(str).str.split("::", n=1, expand=True)
    df = df.copy()
    df["source_dataset"] = parts[0]
    df["local_qid"] = parts[1]
    return df


def load_val_assignments(gmm_dir: Path, k: int, prefix_val: pd.DataFrame) -> pd.DataFrame:
    path = gmm_dir / f"assignments_gmm_val_k{k:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    parts = df["question_id"].astype(str).str.split("::", n=1, expand=True)
    df = df.copy()
    df["source_dataset"] = parts[0]
    df["local_qid"] = parts[1]
    style_map = prefix_val[["style_id", "style_name"]].drop_duplicates()
    df = df.merge(style_map, on="style_id", how="left")
    if df["style_name"].isna().any():
        raise ValueError("val assignments missing style_name after style_id join")
    return df[
        ["source_dataset", "local_qid", "style_name", "style_id", "cluster_id", "question_id"]
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--prefix-dir", type=Path, default=Path("data/scas/modc_prefix_full_sft"))
    p.add_argument("--gmm-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--prefix-mode",
        type=str,
        default="hard",
        choices=("hard", "soft_special"),
    )
    args = p.parse_args()

    train_assign = load_train_assignments(args.gmm_dir, args.k)
    prefix_train = pd.read_parquet(args.prefix_dir / "train.parquet")
    prefix_val = pd.read_parquet(args.prefix_dir / "validation.parquet")
    val_assign = load_val_assignments(args.gmm_dir, args.k, prefix_val)

    train_merged = prefix_train.merge(
        train_assign,
        left_on=["source_dataset", "question_id", "style_name"],
        right_on=["source_dataset", "local_qid", "style_name"],
        how="inner",
        validate="one_to_one",
    )
    if "trace_len" in train_merged.columns:
        lens = train_merged["trace_len"].fillna(0).astype(int).to_numpy()
    else:
        # fallback: assistant message length
        def _alen(m):
            if isinstance(m, str):
                m = json.loads(m)
            return len(m[1]["content"]) if len(m) > 1 else 0

        lens = np.array([_alen(m) for m in train_merged["messages"]], dtype=np.int64)

    rank_map = order_clusters_by_mean_len(train_merged["cluster_id"].to_numpy(), lens)

    train_out, train_stats = build_split(
        prefix_train, train_assign, rank_map, prefix_mode=args.prefix_mode
    )
    val_out, val_stats = build_split(
        prefix_val, val_assign, rank_map, prefix_mode=args.prefix_mode
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "validation.parquet"
    train_out.to_parquet(train_path, index=False)
    val_out.to_parquet(val_path, index=False)

    cue = "[style_i]" if args.prefix_mode == "hard" else "<style_i>"
    payload = {
        "k": args.k,
        "prefix_mode": args.prefix_mode,
        "prompt_format": f"{cue} from GMM K={args.k} on covz z (style_1=shortest cluster)",
        "gmm_dir": str(args.gmm_dir),
        "prefix_dir": str(args.prefix_dir),
        "cluster_rank_map_raw_to_style_index": {str(k): int(v) for k, v in rank_map.items()},
        "n_styles": int(args.k),
        "train": train_stats,
        "validation": val_stats,
    }
    json_out = args.json_out or (args.output_dir / "dataset_meta.json")
    json_out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {train_path} ({len(train_out)}) and {val_path} ({len(val_out)})")


if __name__ == "__main__":
    main()
