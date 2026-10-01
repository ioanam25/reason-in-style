#!/usr/bin/env python3
"""
Assign [style_1] / [style_2] labels from HDBSCAN on z(s) for the full prefix SFT corpus.

Fit HDBSCAN on train-split z embeddings (trace-only disentanglement checkpoint).
Map clusters to style_1 (shorter traces) and style_2 (longer traces).
Assign validation rows via hdbscan.approximate_predict (noise -> nearest centroid).

Prereq: modc_prefix_full_sft parquet + trace-only checkpoint.

Outputs:
  data/scas/cluster_style_assignments/train.parquet
  data/scas/cluster_style_assignments/validation.parquet
  data/scas/cluster_style_assignments/cluster_meta.json
  data/scas/cluster_style_assignments/embeddings_train_z.npz  (cache)

Usage:
  python scripts/discovery/assign_z_cluster_styles.py \
      --checkpoint checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz/last.ckpt \
      --prefix-dir data/scas/modc_prefix_full_sft
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.clustering_utils import (  # noqa: E402
    cluster_silhouette,
    l2_normalize,
    min_cluster_size_candidates,
    run_hdbscan,
)
from scripts.eval.eval_gemini_styles import load_or_extract_embeddings  # noqa: E402

PREFIX_RE = re.compile(r"^\[(DFS|BFS)\]\n", re.IGNORECASE)


def parquet_to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        messages = list(row["messages"])
        trace = messages[1]["content"]
        user = messages[0]["content"]
        question = PREFIX_RE.sub("", user, count=1)
        records.append(
            {
                "question_id": str(row["question_id"]),
                "question": question,
                "trace": trace,
                "trace_len": len(trace),
                "oracle_style_name": str(row.get("style_name", "unknown")),
                "oracle_style_id": int(row.get("style_id", -1)),
            }
        )
    return records


def select_hdbscan_best(
    train_z: np.ndarray,
    max_clusters: int,
    seed: int,
    min_clusters: int = 2,
    max_noise_frac: float = 0.10,
) -> tuple:
    """Pick HDBSCAN run with best silhouette among 2..max_clusters (prefer low noise)."""
    n = len(train_z)
    best = None
    best_sil = -1.0
    sweep = []
    for mcs in min_cluster_size_candidates(n, max_clusters):
        labels, clusterer = run_hdbscan(train_z, mcs, prediction_data=True)
        n_clusters = len(set(labels.tolist()) - {-1})
        n_noise = int((labels == -1).sum())
        sil = cluster_silhouette(train_z, labels)
        row = {
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_frac": float(n_noise / n),
            "silhouette": sil,
        }
        sweep.append(row)
        sil_s = -1.0 if sil is None else sil
        print(
            f"  min_cluster_size={mcs:5d}  clusters={n_clusters:2d}  "
            f"noise={n_noise:6d} ({100 * n_noise / n:5.1f}%)  silhouette={sil_s:.4f}"
            if sil is not None
            else f"  min_cluster_size={mcs:5d}  clusters={n_clusters:2d}  noise={n_noise:6d}  silhouette=n/a"
        )
        if (
            min_clusters <= n_clusters <= max_clusters
            and sil is not None
            and row["noise_frac"] <= max_noise_frac
            and sil > best_sil
        ):
            best_sil = sil
            best = {
                "min_cluster_size": mcs,
                "labels": labels,
                "clusterer": clusterer,
                **row,
            }

    if best is None:
        eligible = [
            r
            for r in sweep
            if min_clusters <= r["n_clusters"] <= max_clusters and r["silhouette"] is not None
        ]
        if not eligible:
            raise RuntimeError("HDBSCAN found no usable clusters; try lowering min_cluster_size sweep.")
        pick = max(eligible, key=lambda r: (r["silhouette"] or -1.0, -r["noise_frac"]))
        labels, clusterer = run_hdbscan(train_z, pick["min_cluster_size"], prediction_data=True)
        best = {
            "min_cluster_size": pick["min_cluster_size"],
            "labels": labels,
            "clusterer": clusterer,
            **pick,
        }

    return best, sweep


def select_hdbscan_k2(train_z: np.ndarray, max_clusters: int, seed: int) -> tuple:
    """Backward-compatible wrapper: prefer exactly 2 clusters when available."""
    return select_hdbscan_best(train_z, max_clusters, seed, min_clusters=2, max_noise_frac=1.0)


def resolve_noise_labels(labels: np.ndarray, z: np.ndarray, centroids: dict[int, np.ndarray]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).copy()
    z_norm = l2_normalize(z)
    for i in range(len(labels)):
        if labels[i] < 0:
            labels[i] = assign_nearest_centroid(z_norm[i], centroids)
    return labels


def cluster_to_style_names(
    labels: np.ndarray,
    trace_lens: np.ndarray,
) -> tuple[dict[int, str], dict[int, int], list[str]]:
    """Map HDBSCAN cluster ids -> style_1..style_K ordered by mean trace length (short first)."""
    clusters = sorted(set(labels.tolist()) - {-1})
    if len(clusters) < 2:
        raise ValueError(f"Expected >=2 clusters, got {clusters}")

    mean_lens = {}
    for cid in clusters:
        mask = labels == cid
        mean_lens[cid] = float(np.mean(trace_lens[mask]))

    ordered = sorted(clusters, key=lambda c: mean_lens[c])
    name_map = {cid: f"style_{i + 1}" for i, cid in enumerate(ordered)}
    id_map = {cid: i for i, cid in enumerate(ordered)}
    style_names = [name_map[cid] for cid in ordered]
    return name_map, id_map, style_names


def compute_centroids(z: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    z_norm = l2_normalize(z)
    centroids = {}
    for cid in set(labels.tolist()) - {-1}:
        mask = labels == cid
        centroids[int(cid)] = z_norm[mask].mean(axis=0)
    return centroids


def assign_nearest_centroid(z_norm_row: np.ndarray, centroids: dict[int, np.ndarray]) -> int:
    best_c, best_sim = -1, -2.0
    for cid, cen in centroids.items():
        sim = float(np.dot(z_norm_row, cen))
        if sim > best_sim:
            best_sim = sim
            best_c = cid
    return best_c


def predict_labels(
    clusterer,
    z: np.ndarray,
    train_centroids: dict[int, np.ndarray],
) -> np.ndarray:
    """Assign points to train clusters (nearest centroid; approximate_predict if available)."""
    z_norm = l2_normalize(z)
    try:
        import hdbscan

        if getattr(clusterer, "prediction_data_", False):
            labels, _strengths = hdbscan.approximate_predict(clusterer, z_norm)
            labels = np.asarray(labels, dtype=np.int64)
            for i in range(len(labels)):
                if labels[i] < 0:
                    labels[i] = assign_nearest_centroid(z_norm[i], train_centroids)
            return labels
    except (AttributeError, ValueError):
        pass

    return np.array(
        [assign_nearest_centroid(z_norm[i], train_centroids) for i in range(len(z))],
        dtype=np.int64,
    )


def build_assignment_df(
    records: list[dict],
    labels: np.ndarray,
    name_map: dict[int, str],
    id_map: dict[int, int],
) -> pd.DataFrame:
    rows = []
    for rec, lab in zip(records, labels):
        lab = int(lab)
        style_name = name_map[lab]
        rows.append(
            {
                "question_id": rec["question_id"],
                "trace_len": rec["trace_len"],
                "oracle_style_name": rec["oracle_style_name"],
                "oracle_style_id": rec["oracle_style_id"],
                "cluster_id": lab,
                "style_name": style_name,
                "style_id": id_map[lab],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Assign style_1/style_2 from z(s) HDBSCAN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz/last.ckpt",
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path("data/scas/modc_prefix_full_sft"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scas/cluster_style_assignments"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument(
        "--max-noise-frac",
        type=float,
        default=0.10,
        help="Prefer HDBSCAN runs with at most this noise fraction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Subsample train split to this many rows before embedding/HDBSCAN (0 = all)",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip validation embedding and assignment (train clustering only)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Skip sweep; fit HDBSCAN once at this min_cluster_size (use for fixed K=2)",
    )
    parser.add_argument(
        "--require-k",
        type=int,
        default=None,
        help="If set, fail unless HDBSCAN finds exactly this many clusters (e.g. 2)",
    )
    parser.add_argument("--reextract-embeddings", action="store_true")
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Skip train HDBSCAN; load train.parquet + cached train z, assign validation only",
    )
    args = parser.parse_args()

    import torch
    from src.module import DisentangledLightningModule

    train_path = args.prefix_dir / "train.parquet"
    val_path = args.prefix_dir / "validation.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"Missing parquet in {args.prefix_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading train parquet: {train_path}")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    train_records = parquet_to_records(train_df)
    val_records = parquet_to_records(val_df)
    if args.max_records > 0 and len(train_records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(train_records), size=args.max_records, replace=False)
        train_records = [train_records[int(i)] for i in pick]
        print(f"  Subsampled train to {len(train_records)} records (max_records={args.max_records})")
    print(f"  train={len(train_records)}  val={len(val_records)}")

    train_out = args.output_dir / "train.parquet"
    val_out = args.output_dir / "validation.parquet"
    emb_cache = args.output_dir / "embeddings_train_z.npz"
    val_cache = args.output_dir / "embeddings_val_z.npz"

    if args.val_only:
        if not train_out.is_file():
            raise FileNotFoundError(f"--val-only requires {train_out}")
        print("\n--val-only: loading existing train assignments...")
        train_assign = pd.read_parquet(train_out)
        if len(train_assign) != len(train_records):
            raise ValueError("train.parquet row count does not match prefix train split")
        train_labels = train_assign["cluster_id"].to_numpy(dtype=np.int64)
        name_map = {
            int(cid): str(train_assign.loc[train_assign["cluster_id"] == cid, "style_name"].iloc[0])
            for cid in sorted(set(train_labels.tolist()))
        }
        id_map = {
            int(cid): int(train_assign.loc[train_assign["cluster_id"] == cid, "style_id"].iloc[0])
            for cid in sorted(set(train_labels.tolist()))
        }
        style_names = sorted(set(name_map.values()), key=lambda s: int(s.split("_")[1]))
        best = {"min_cluster_size": args.min_cluster_size, "val_only": True}
        sweep = []
        clusterer = None
        z_val = None

        if emb_cache.is_file() and val_cache.is_file():
            print("Using cached train/val z embeddings (no checkpoint load)...")
            z_train = np.load(emb_cache)["z_s"]
            z_val = np.load(val_cache)["z_s"]
        else:
            print(f"\nLoading checkpoint: {args.checkpoint}")
            module = DisentangledLightningModule.load_from_checkpoint(
                args.checkpoint, map_location="cpu", strict=False,
            )
            module.eval()
            max_len = module.hparams.get("max_len", 4096)
            if emb_cache.is_file():
                z_train = np.load(emb_cache)["z_s"]
                print(f"  Loaded cached train z: {emb_cache}")
            else:
                _c_train, z_train = load_or_extract_embeddings(
                    module, train_records, args.batch_size, device, max_len, emb_cache,
                )
            if val_cache.is_file():
                z_val = np.load(val_cache)["z_s"]
                print(f"  Loaded cached validation z: {val_cache}")
            else:
                _c_val, z_val = load_or_extract_embeddings(
                    module, val_records, args.batch_size, device, max_len, val_cache,
                )

        centroids = compute_centroids(z_train, train_labels)

        print("\nAssigning validation clusters...")
        if z_val is None:
            raise RuntimeError("validation z embeddings missing")
        val_labels = predict_labels(clusterer, z_val, centroids)
        val_assign = build_assignment_df(val_records, val_labels, name_map, id_map)
        val_assign.to_parquet(val_out, index=False)
        print(f"Wrote {val_out}")

        def summarize(assign_df: pd.DataFrame, split: str) -> dict:
            c = Counter(assign_df["style_name"])
            oc = Counter(zip(assign_df["oracle_style_name"], assign_df["style_name"]))
            return {
                "split": split,
                "n": int(len(assign_df)),
                "style_counts": dict(c),
                "mean_trace_len_by_style": {
                    sn: float(assign_df.loc[assign_df["style_name"] == sn, "trace_len"].mean())
                    for sn in sorted(c.keys())
                },
                "oracle_vs_cluster_crosstab": {f"{a}|{b}": int(n) for (a, b), n in oc.items()},
            }

        meta = {
            "checkpoint": args.checkpoint,
            "prefix_dir": str(args.prefix_dir),
            "method": "hdbscan_z_train_fit_nearest_centroid_val",
            "cluster_to_style": {str(k): v for k, v in name_map.items()},
            "style_id_map": id_map,
            "style_names": style_names,
            "n_clusters": len(style_names),
            "selected_hdbscan": best,
            "hdbscan_sweep": sweep,
            "train_summary": summarize(train_assign, "train"),
            "val_summary": summarize(val_assign, "validation"),
            "train_style_proportions": {
                sn: train_assign["style_name"].value_counts()[sn] / len(train_assign)
                for sn in style_names
            },
            "note": "style_i ordered by mean trace length (style_1 = shortest cluster)",
        }
        meta_path = args.output_dir / "cluster_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\nWrote {meta_path}")
        print("\nTrain style counts:", meta["train_summary"]["style_counts"])
        print("Val style counts:", meta["val_summary"]["style_counts"])
        return

    print(f"\nLoading checkpoint: {args.checkpoint}")
    module = DisentangledLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=False,
    )
    module.eval()
    max_len = module.hparams.get("max_len", 4096)

    emb_cache = args.output_dir / "embeddings_train_z.npz"
    if args.reextract_embeddings and emb_cache.exists():
        emb_cache.unlink()

    print("\nExtracting z(s) for train...")
    _c_train, z_train = load_or_extract_embeddings(
        module,
        train_records,
        args.batch_size,
        device,
        max_len,
        emb_cache,
    )

    if args.min_cluster_size is not None:
        print(f"\nHDBSCAN on train z(s) (fixed min_cluster_size={args.min_cluster_size})...")
        train_labels, clusterer = run_hdbscan(
            z_train, args.min_cluster_size, prediction_data=True,
        )
        n_clusters = len(set(train_labels.tolist()) - {-1})
        n_noise = int((train_labels == -1).sum())
        sil = cluster_silhouette(z_train, train_labels)
        best = {
            "min_cluster_size": args.min_cluster_size,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_frac": float(n_noise / len(z_train)),
            "silhouette": sil,
            "fixed": True,
        }
        sweep = []
        print(
            f"  clusters={n_clusters}  noise={n_noise}  silhouette={sil}"
            if sil is not None
            else f"  clusters={n_clusters}  noise={n_noise}  silhouette=n/a"
        )
        req_k = args.require_k if args.require_k is not None else (
            2 if args.min_cluster_size is not None and args.max_clusters == 2 else None
        )
        if req_k is not None and n_clusters != req_k:
            raise RuntimeError(
                f"HDBSCAN found {n_clusters} clusters, expected {req_k} "
                f"(min_cluster_size={args.min_cluster_size})"
            )
    else:
        print(f"\nHDBSCAN sweep on train z(s) (target: {args.min_clusters}-{args.max_clusters} clusters)...")
        best, sweep = select_hdbscan_best(
            z_train,
            args.max_clusters,
            args.seed,
            min_clusters=args.min_clusters,
            max_noise_frac=args.max_noise_frac,
        )
        train_labels = best.pop("labels")
        clusterer = best.pop("clusterer")

    trace_lens_train = np.array([r["trace_len"] for r in train_records])
    centroids = compute_centroids(z_train, train_labels)
    train_labels = resolve_noise_labels(train_labels, z_train, centroids)
    name_map, id_map, style_names = cluster_to_style_names(train_labels, trace_lens_train)

    print(f"\nSelected min_cluster_size={best['min_cluster_size']}  clusters={best.get('n_clusters')}  silhouette={best.get('silhouette')}")
    print(f"  cluster -> style mapping (by mean trace length): {name_map}")

    train_assign = build_assignment_df(train_records, train_labels, name_map, id_map)
    train_out = args.output_dir / "train.parquet"
    train_assign.to_parquet(train_out, index=False)
    print(f"Wrote {train_out}")

    if args.skip_val:
        val_assign = None
        print("\n--skip-val: skipping validation embedding and assignment")
    else:
        print("\nExtracting z(s) for validation + approximate_predict...")
        val_cache = args.output_dir / "embeddings_val_z.npz"
        _c_val, z_val = load_or_extract_embeddings(
            module, val_records, args.batch_size, device, max_len, val_cache,
        )
        val_labels = predict_labels(clusterer, z_val, centroids)
        val_assign = build_assignment_df(val_records, val_labels, name_map, id_map)
        val_out = args.output_dir / "validation.parquet"
        val_assign.to_parquet(val_out, index=False)
        print(f"Wrote {val_out}")

    def summarize(assign_df: pd.DataFrame, split: str) -> dict:
        c = Counter(assign_df["style_name"])
        oc = Counter(
            zip(assign_df["oracle_style_name"], assign_df["style_name"])
        )
        return {
            "split": split,
            "n": int(len(assign_df)),
            "style_counts": dict(c),
            "mean_trace_len_by_style": {
                sn: float(assign_df.loc[assign_df["style_name"] == sn, "trace_len"].mean())
                for sn in sorted(c.keys())
            },
            "oracle_vs_cluster_crosstab": {f"{a}|{b}": int(n) for (a, b), n in oc.items()},
        }

    meta = {
        "checkpoint": args.checkpoint,
        "prefix_dir": str(args.prefix_dir),
        "method": "hdbscan_z_train_fit_approx_predict_val",
        "cluster_to_style": {str(k): v for k, v in name_map.items()},
        "style_id_map": id_map,
        "style_names": style_names,
        "n_clusters": len(style_names),
        "selected_hdbscan": best,
        "hdbscan_sweep": sweep,
        "train_summary": summarize(train_assign, "train"),
        "train_style_proportions": {
            sn: float(train_assign["style_name"].value_counts()[sn] / len(train_assign))
            for sn in style_names
        },
        "max_records": args.max_records,
        "skip_val": args.skip_val,
        "note": "style_i ordered by mean trace length (style_1 = shortest cluster)",
    }
    if val_assign is not None:
        meta["val_summary"] = summarize(val_assign, "validation")
    meta_path = args.output_dir / "cluster_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta_path}")
    print("\nTrain style counts:", meta["train_summary"]["style_counts"])
    print("Train style proportions:", meta["train_style_proportions"])
    if val_assign is not None:
        print("Val style counts:", meta["val_summary"]["style_counts"])


if __name__ == "__main__":
    main()
