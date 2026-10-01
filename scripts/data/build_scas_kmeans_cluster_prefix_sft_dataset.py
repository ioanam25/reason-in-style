#!/usr/bin/env python3
"""Build ModC-style prefix SFT data using k-means cluster labels (style_1..style_K)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.clustering_utils import l2_normalize  # noqa: E402

from scripts.scas_style_prefix import (  # noqa: E402
    strip_style_prefix,
    user_content_with_style,
)

PREFIX_RE = re.compile(r"^\[[^\]]+\]\n")  # kept for older call sites / docs


def cluster_id_to_style_name(cluster_id: int) -> str:
    return f"style_{int(cluster_id) + 1}"


def order_clusters_by_mean_len(cluster_ids: np.ndarray, trace_lens: np.ndarray) -> dict[int, int]:
    """Map raw cluster_id -> style index (0=shortest mean len) for style_1 naming."""
    buckets: dict[int, list[int]] = {}
    for cid, tl in zip(cluster_ids, trace_lens):
        buckets.setdefault(int(cid), []).append(int(tl))
    ranked = sorted(buckets.items(), key=lambda kv: float(np.mean(kv[1])))
    return {int(cid): rank for rank, (cid, _) in enumerate(ranked)}


def assign_val_with_centers(z_val: np.ndarray, centers: np.ndarray) -> np.ndarray:
    z = l2_normalize(z_val)
    c = l2_normalize(centers)
    # nearest centroid (euclidean on unit sphere ~ max cosine)
    sims = z @ c.T
    return sims.argmax(axis=1).astype(np.int64)


def load_train_assignments(kmeans_dir: Path, k: int) -> pd.DataFrame:
    path = kmeans_dir / f"assignments_kmeans_k{k:02d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    parts = df["question_id"].astype(str).str.split("::", n=1, expand=True)
    df = df.copy()
    df["source_dataset"] = parts[0]
    df["local_qid"] = parts[1]
    return df


def load_val_assignments(
    kmeans_dir: Path,
    k: int,
    val_emb_npz: Path,
    val_records: pd.DataFrame,
) -> pd.DataFrame:
    centers = np.load(kmeans_dir / f"centers_kmeans_k{k:02d}.npy")
    cached = np.load(val_emb_npz, allow_pickle=True)
    z_val = np.asarray(cached["z_s"], dtype=np.float64)
    qids = [str(q) for q in cached["question_id"]]
    style_ids = np.asarray(cached["style_id"])

    emb_df = pd.DataFrame({"question_id": qids, "style_id": style_ids, "row_idx": np.arange(len(qids))})
    emb_parts = emb_df["question_id"].str.split("::", n=1, expand=True)
    emb_df["source_dataset"] = emb_parts[0]
    emb_df["local_qid"] = emb_parts[1]
    style_map = val_records[["style_id", "style_name"]].drop_duplicates()
    emb_df = emb_df.merge(style_map, on="style_id", how="left")

    labels = assign_val_with_centers(z_val, centers)
    emb_df["cluster_id"] = labels

    return emb_df[
        ["source_dataset", "local_qid", "style_name", "style_id", "cluster_id", "question_id"]
    ]


def build_split(
    prefix_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    rank_map: dict[int, int],
    *,
    prefix_mode: str = "hard",
) -> tuple[pd.DataFrame, dict]:
    merged = prefix_df.merge(
        assign_df,
        left_on=["source_dataset", "question_id", "style_name"],
        right_on=["source_dataset", "local_qid", "style_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(prefix_df):
        raise ValueError(f"join dropped rows: {len(merged)} vs {len(prefix_df)}")

    style_names = []
    style_ids = []
    cluster_styles = []
    new_messages = []
    msg_col = "messages_x" if "messages_x" in merged.columns else "messages"
    for _, row in merged.iterrows():
        raw_cid = int(row["cluster_id"])
        style_idx = rank_map[raw_cid]
        style_name = cluster_id_to_style_name(style_idx)
        messages = row[msg_col]
        if isinstance(messages, str):
            messages = json.loads(messages)
        messages = list(messages)
        question = strip_style_prefix(messages[0]["content"])
        new_messages.append(
            [
                {
                    "role": "user",
                    "content": user_content_with_style(style_name, question, prefix_mode),
                },
                {"role": "assistant", "content": messages[1]["content"]},
            ]
        )
        style_names.append(style_name)
        style_ids.append(style_idx)
        cluster_styles.append(style_name)

    out = prefix_df.loc[merged.index].copy()
    out["messages"] = new_messages
    out["style_name"] = style_names
    out["style_id"] = style_ids
    out["cluster_id"] = merged["cluster_id"].values
    out["cluster_style_name"] = cluster_styles
    teacher_col = "style_name_y" if "style_name_y" in merged.columns else "style_name"
    out["oracle_style_name"] = merged[teacher_col].values

    stats = {
        "rows": int(len(out)),
        "cluster_style_counts": dict(Counter(cluster_styles)),
        "raw_cluster_counts": dict(Counter(int(x) for x in merged["cluster_id"])),
        "prefix_mode": prefix_mode,
    }
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--prefix-dir", type=Path, default=Path("data/scas/modc_prefix_full_sft"))
    p.add_argument("--kmeans-dir", type=Path, default=Path("data/scas/cluster_kmeans_sweep-qwen3-4b"))
    p.add_argument("--val-embeddings-npz", type=Path, default=Path("data/scas/cluster_gmm_assignments-qwen3-4b/embeddings_val_z.npz"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--prefix-mode",
        type=str,
        default="hard",
        choices=("hard", "soft_special"),
        help="hard=[style_i]; soft_special=<style_i> dedicated special token string",
    )
    args = p.parse_args()

    train_assign = load_train_assignments(args.kmeans_dir, args.k)

    prefix_train = pd.read_parquet(args.prefix_dir / "train.parquet")
    prefix_val = pd.read_parquet(args.prefix_dir / "validation.parquet")
    val_assign = load_val_assignments(args.kmeans_dir, args.k, args.val_embeddings_npz, prefix_val)

    train_merged = prefix_train.merge(
        train_assign,
        left_on=["source_dataset", "question_id", "style_name"],
        right_on=["source_dataset", "local_qid", "style_name"],
        how="inner",
        validate="one_to_one",
    )
    rank_map = order_clusters_by_mean_len(
        train_merged["cluster_id"].to_numpy(),
        train_merged["trace_len"].fillna(0).astype(int).to_numpy(),
    )

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
        "prompt_format": f"{cue} from k-means K={args.k} on AE z (style_1=shortest cluster)",
        "kmeans_dir": str(args.kmeans_dir),
        "prefix_dir": str(args.prefix_dir),
        "cluster_rank_map_raw_to_style_index": {str(k): int(v) for k, v in rank_map.items()},
        "train": train_stats,
        "validation": val_stats,
    }
    json_out = args.json_out or (args.output_dir / "dataset_meta.json")
    json_out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
