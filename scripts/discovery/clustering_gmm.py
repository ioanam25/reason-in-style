#!/usr/bin/env python3
"""
GMM clustering on z(s) after trace-only (autoencoder) training.

Uses the same HF train split as disentanglement training. That split is
re-partitioned by question_id into gmm-train / gmm-val to pick the number
of components (lowest validation BIC). The final GMM is refit on all
autoencoder-train rows with the selected K and cluster labels are written
as style_1..style_K (ordered by mean trace length).

Outputs:
  <output-dir>/train.parquet
  <output-dir>/cluster_meta.json
  <output-dir>/embeddings_train_z.npz

Usage:
  python scripts/discovery/clustering_gmm.py \\
      --checkpoint checkpoints/scas-trace-only-qwen3-4b-vicreg-b8-covz/last.ckpt \\
      --hf-dataset scas_traces_dataset \\
      --output-dir data/scas/cluster_gmm_assignments-qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import split_records_by_question  # noqa: E402
from scripts.discovery.assign_z_cluster_styles import (  # noqa: E402
    build_assignment_df,
    cluster_to_style_names,
)
from scripts.clustering_utils import l2_normalize  # noqa: E402
from scripts.eval.eval_gemini_styles import (  # noqa: E402
    load_gemini_with_styles,
    load_or_extract_embeddings,
)
from scripts.repo_paths import CONFIG_SCAS  # noqa: E402


def enrich_records(records: list[dict]) -> list[dict]:
    for rec in records:
        rec["trace_len"] = len(rec["trace"])
        rec.setdefault("oracle_style_name", str(rec.get("style_name", "unknown")))
        rec.setdefault("oracle_style_id", int(rec.get("style_id", -1)))
    return records


def fit_gmm(z: np.ndarray, n_components: int, seed: int) -> GaussianMixture:
    z = np.asarray(z, dtype=np.float64)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=seed,
        n_init=5,
        max_iter=300,
        reg_covar=1e-3,
    )
    gmm.fit(z)
    return gmm


def val_silhouette(z_val: np.ndarray, labels: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2 or len(labels) < 2:
        return None
    return float(silhouette_score(z_val, labels, metric="euclidean"))


def eval_gmm_k(
    k: int,
    z_gmm_train: np.ndarray,
    z_gmm_val: np.ndarray,
    seed: int,
    z_all: np.ndarray | None = None,
) -> dict:
    try:
        gmm = fit_gmm(z_gmm_train, k, seed)
        z_train = np.asarray(z_gmm_train, dtype=np.float64)
        z_val = np.asarray(z_gmm_val, dtype=np.float64)
        val_labels = gmm.predict(z_val)
        train_labels = gmm.predict(z_train)
        sil = val_silhouette(z_gmm_val, val_labels)
        train_ll_mean = float(gmm.score(z_train))
        val_ll_mean = float(gmm.score(z_val))
        result = {
            "n_components": k,
            "failed": False,
            "train_bic": float(gmm.bic(z_train)),
            "val_bic": float(gmm.bic(z_val)),
            "train_log_likelihood_mean": train_ll_mean,
            "val_log_likelihood_mean": val_ll_mean,
            "train_log_likelihood_total": train_ll_mean * len(z_train),
            "val_log_likelihood_total": val_ll_mean * len(z_val),
            "val_silhouette": sil,
            "cluster_counts_gmm_train": {
                int(c): int(n) for c, n in sorted(Counter(train_labels).items())
            },
            "cluster_counts_gmm_val": {
                int(c): int(n) for c, n in sorted(Counter(val_labels).items())
            },
        }
        if z_all is not None:
            all_labels = gmm.predict(np.asarray(z_all, dtype=np.float64))
            result["cluster_counts_full_train"] = {
                int(c): int(n) for c, n in sorted(Counter(all_labels).items())
            }
        return result
    except ValueError as exc:
        return {
            "n_components": k,
            "failed": True,
            "error": str(exc),
            "train_bic": float("inf"),
            "val_bic": float("inf"),
            "train_log_likelihood_mean": float("-inf"),
            "val_log_likelihood_mean": float("-inf"),
            "train_log_likelihood_total": float("-inf"),
            "val_log_likelihood_total": float("-inf"),
            "val_silhouette": None,
        }


def sweep_gmm_k(
    z_gmm_train: np.ndarray,
    z_gmm_val: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    seed: int,
    n_jobs: int = 1,
    z_all: np.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    """Fit GMM on gmm-train for each K in parallel; pick K with lowest validation BIC."""
    ks = list(range(min_clusters, max_clusters + 1))
    if n_jobs == 1 or len(ks) == 1:
        sweep = [eval_gmm_k(k, z_gmm_train, z_gmm_val, seed, z_all) for k in ks]
    else:
        from joblib import Parallel, delayed

        sweep = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(eval_gmm_k)(k, z_gmm_train, z_gmm_val, seed, z_all) for k in ks
        )

    for row in sorted(sweep, key=lambda r: r["n_components"]):
        if row.get("failed"):
            print(f"  K={row['n_components']:2d}  FAILED: {row.get('error', 'unknown')[:120]}")
            continue
        sil_s = "n/a" if row["val_silhouette"] is None else f"{row['val_silhouette']:.4f}"
        counts = row.get("cluster_counts_full_train") or row.get("cluster_counts_gmm_train", {})
        print(
            f"  K={row['n_components']:2d}  "
            f"val_ll={row['val_log_likelihood_total']:14.1f}  "
            f"train_ll={row['train_log_likelihood_total']:14.1f}  "
            f"val_bic={row['val_bic']:12.1f}  "
            f"val_silhouette={sil_s}  "
            f"cluster_sizes={counts}"
        )

    ok = [r for r in sweep if not r.get("failed")]
    if not ok:
        raise RuntimeError("GMM sweep: all K values failed to fit")
    best = min(ok, key=lambda r: r["val_bic"])
    return best, sweep


def summarize_assignments(assign_df: pd.DataFrame, split: str) -> dict:
    style_counts = Counter(assign_df["style_name"])
    oc = Counter(zip(assign_df["oracle_style_name"], assign_df["style_name"]))
    return {
        "split": split,
        "n": int(len(assign_df)),
        "style_counts": dict(style_counts),
        "mean_trace_len_by_style": {
            sn: float(assign_df.loc[assign_df["style_name"] == sn, "trace_len"].mean())
            for sn in sorted(style_counts.keys())
        },
        "oracle_vs_cluster_crosstab": {f"{a}|{b}": int(n) for (a, b), n in oc.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GMM clustering on z(s) with val BIC model selection")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="scas_traces_dataset",
        help="HF dataset used for trace-only training (train split only)",
    )
    parser.add_argument(
        "--raw-json",
        type=str,
        default=str(CONFIG_SCAS / "scas_traces_dataset.json"),
        help="Optional style join JSON (unused when HF rows already have style_id)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gmm-val-fraction",
        type=float,
        default=0.1,
        help="Fraction of question_ids held out for GMM K selection (by question_id)",
    )
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument(
        "--gmm-jobs",
        type=int,
        default=0,
        help="Parallel workers for K sweep (0 = use all CPUs on the node)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Subsample autoencoder train rows before embedding (0 = all)",
    )
    parser.add_argument("--reextract-embeddings", action="store_true")
    parser.add_argument(
        "--gmm-only",
        action="store_true",
        help="Skip embedding extraction; require embeddings_train_z.npz in output-dir",
    )
    parser.add_argument(
        "--skip-val-split",
        action="store_true",
        help="Skip HF validation split assignment (train GMM labels only)",
    )
    parser.add_argument(
        "--assign-val-split-only",
        action="store_true",
        help="Only assign HF validation using existing train GMM outputs",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=None,
        help="NPZ with z_s aligned to train records (default: output-dir/embeddings_train_z.npz)",
    )
    args = parser.parse_args()

    if args.assign_val_split_only:
        from scripts.discovery.assign_gmm_validation_split import assign_hf_validation_split

        assign_hf_validation_split(
            assignments_dir=args.output_dir,
            checkpoint=args.checkpoint,
            hf_dataset=args.hf_dataset,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            reextract=args.reextract_embeddings,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = args.embeddings_npz or (args.output_dir / "embeddings_train_z.npz")

    print(f"Loading autoencoder train split from {args.hf_dataset}...")
    records = load_gemini_with_styles(args.hf_dataset, args.raw_json, split="train")
    records = enrich_records(records)
    if args.max_records > 0 and len(records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[int(i)] for i in pick]
        print(f"  Subsampled to {len(records)} records (max_records={args.max_records})")
    print(f"  n_records={len(records)}")

    gmm_train_records, gmm_val_records = split_records_by_question(
        records, args.gmm_val_fraction,
    )
    print(
        f"  GMM model-selection split: train={len(gmm_train_records)}  "
        f"val={len(gmm_val_records)}  (fraction={args.gmm_val_fraction})"
    )

    if args.gmm_only:
        if not emb_cache.is_file():
            raise FileNotFoundError(f"--gmm-only requires {emb_cache}")
        checkpoint = args.checkpoint
        manifest_path = args.output_dir / "embedding_manifest.json"
        if checkpoint is None and manifest_path.is_file():
            checkpoint = json.loads(manifest_path.read_text()).get("checkpoint")
        print(f"\nLoading cached embeddings: {emb_cache}")
        cached = np.load(emb_cache, allow_pickle=False)
        if len(cached["z_s"]) != len(records):
            raise ValueError(
                f"Embedding count {len(cached['z_s'])} != record count {len(records)}"
            )
        cached_qids = [str(q) for q in cached["question_id"]]
        record_qids = [r["question_id"] for r in records]
        if cached_qids != record_qids:
            raise ValueError("question_id order in NPZ does not match loaded records")
        z_all = l2_normalize(cached["z_s"])
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --gmm-only")
        import torch
        from src.module import DisentangledLightningModule

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"\nLoading checkpoint: {args.checkpoint}")
        module = DisentangledLightningModule.load_from_checkpoint(
            args.checkpoint, map_location="cpu", strict=False,
        )
        module.eval()
        max_len = module.hparams.get("max_len", 4096)

        if args.reextract_embeddings and emb_cache.exists():
            emb_cache.unlink()

        print("\nExtracting z(s) for full autoencoder train set...")
        _c_all, z_all = load_or_extract_embeddings(
            module,
            records,
            args.batch_size,
            device,
            max_len,
            emb_cache,
        )
        z_all = l2_normalize(z_all)

    checkpoint = args.checkpoint
    if checkpoint is None:
        manifest_path = args.output_dir / "embedding_manifest.json"
        if manifest_path.is_file():
            checkpoint = json.loads(manifest_path.read_text()).get("checkpoint")

    record_index = {id(rec): i for i, rec in enumerate(records)}
    gmm_train_idx = np.array([record_index[id(r)] for r in gmm_train_records])
    gmm_val_idx = np.array([record_index[id(r)] for r in gmm_val_records])
    z_gmm_train = z_all[gmm_train_idx]
    z_gmm_val = z_all[gmm_val_idx]

    import os

    gmm_jobs = args.gmm_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    n_ks = args.max_clusters - args.min_clusters + 1
    gmm_jobs = max(1, min(gmm_jobs, n_ks, 8))
    print(
        f"\nSweeping GMM K in [{args.min_clusters}, {args.max_clusters}] "
        f"(select by lowest val BIC, n_jobs={gmm_jobs})..."
    )
    best, sweep = sweep_gmm_k(
        z_gmm_train,
        z_gmm_val,
        args.min_clusters,
        args.max_clusters,
        args.seed,
        n_jobs=gmm_jobs,
        z_all=z_all,
    )
    selected_k = int(best["n_components"])
    print(f"\nSelected K={selected_k} (val_bic={best['val_bic']:.1f})")

    print(f"\nRefitting GMM on all {len(records)} autoencoder-train rows...")
    final_gmm = fit_gmm(z_all, selected_k, args.seed)
    labels = final_gmm.predict(z_all)

    trace_lens = np.array([r["trace_len"] for r in records])
    name_map, id_map, style_names = cluster_to_style_names(labels, trace_lens)
    assign_df = build_assignment_df(records, labels, name_map, id_map)

    train_out = args.output_dir / "train.parquet"
    assign_df.to_parquet(train_out, index=False)
    print(f"Wrote {train_out}")

    meta = {
        "checkpoint": checkpoint,
        "hf_dataset": args.hf_dataset,
        "method": "gmm_z_bic_model_selection",
        "gmm_val_fraction": args.gmm_val_fraction,
        "n_autoencoder_train": len(records),
        "n_gmm_train": len(gmm_train_records),
        "n_gmm_val": len(gmm_val_records),
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "selected_k": selected_k,
        "gmm_sweep": sweep,
        "selected_gmm": best,
        "final_gmm": {
            "n_components": selected_k,
            "train_bic": float(final_gmm.bic(z_all)),
            "train_log_likelihood_mean": float(final_gmm.score(z_all)),
            "train_log_likelihood_total": float(final_gmm.score(z_all)) * len(z_all),
            "converged": bool(final_gmm.converged_),
            "n_iter": int(final_gmm.n_iter_),
        },
        "cluster_to_style": {str(k): v for k, v in name_map.items()},
        "style_id_map": id_map,
        "style_names": style_names,
        "train_summary": summarize_assignments(assign_df, "autoencoder_train"),
        "train_style_proportions": {
            sn: assign_df["style_name"].value_counts()[sn] / len(assign_df)
            for sn in style_names
        },
        "note": "style_i ordered by mean trace length (style_1 = shortest cluster)",
    }
    meta_path = args.output_dir / "cluster_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")
    print("\nStyle counts:", meta["train_summary"]["style_counts"])
    if args.skip_val_split:
        print("\nSkipped HF validation assignment (--skip-val-split).")
        print(f"  Run: sbatch scripts/discovery/assign-gmm-validation-scas.slurm <size>")
    else:
        print(
            "\nNext: assign HF validation labels (GPU, uses train-fitted GMM):\n"
            f"  sbatch scripts/discovery/assign-gmm-validation-scas.slurm"
        )


if __name__ == "__main__":
    main()
