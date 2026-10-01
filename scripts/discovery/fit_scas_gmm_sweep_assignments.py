#!/usr/bin/env python3
"""Fit SCAS AE-z GMMs for K=2..10 on the full train set and save assignments.

Uses the same GMM hyperparameters as scripts/discovery/clustering_gmm.py. The original
pipeline only persisted labels for the BIC-selected K; this writes per-K
assignment parquets (train + val) analogous to cluster_kmeans_sweep.

Example:
  ./.venv/bin/python scripts/discovery/fit_scas_gmm_sweep_assignments.py
  ./.venv/bin/python scripts/discovery/fit_scas_gmm_sweep_assignments.py --k-min 2 --k-max 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[2]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def fit_gmm(z: np.ndarray, n_components: int, seed: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=seed,
        n_init=5,
        max_iter=300,
        reg_covar=1e-3,
    )
    gmm.fit(np.asarray(z, dtype=np.float64))
    return gmm


def cluster_to_style_names(
    labels: np.ndarray, lens: np.ndarray, k: int
) -> dict[int, str]:
    """style_1 = shortest mean length (same convention as clustering_gmm.py).

    Empty components (no assigned points) get +inf mean length so they rank last.
    """
    means = {}
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            means[c] = float(np.mean(lens[mask]))
        else:
            means[c] = float("inf")
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    return {cid: f"style_{i + 1}" for i, (cid, _) in enumerate(ranked)}


def load_meta_frame(gmm_dir: Path, emb) -> pd.DataFrame:
    """Align question_id / teacher / length with embedding rows.

    Prefer ``train.parquet`` from a prior clustering dir. If absent, build meta
    from the embedding NPZ alone (works right after ``extract-scas-z-8gpu``).
    """
    train_pq = gmm_dir / "train.parquet"
    if not train_pq.is_file():
        qids = np.asarray(emb["question_id"]).astype(str)
        sids = np.asarray(emb["style_id"]).astype(np.int32)
        lens = (
            np.asarray(emb["trace_len"]).astype(np.float64)
            if "trace_len" in emb.files
            else np.full(len(qids), -1.0)
        )
        names = (
            np.asarray(emb["style_name"]).astype(str)
            if "style_name" in emb.files
            else sids.astype(str)
        )
        return pd.DataFrame(
            {
                "question_id": qids,
                "join_style_id": sids,
                "emb_style_id": sids,
                "oracle_style_id": sids,
                "oracle_style_name": names,
                "trace_len": lens,
            }
        )

    saved = pd.read_parquet(train_pq).copy()
    saved["question_id"] = saved["question_id"].astype(str)
    saved["join_style_id"] = saved["oracle_style_id"].astype(np.int32)
    base = pd.DataFrame(
        {
            "question_id": np.asarray(emb["question_id"]).astype(str),
            "join_style_id": np.asarray(emb["style_id"]).astype(np.int32),
            "emb_style_id": np.asarray(emb["style_id"]).astype(np.int32),
        }
    )
    cols = ["question_id", "join_style_id", "oracle_style_name", "oracle_style_id"]
    if "trace_len" in saved.columns:
        cols.append("trace_len")
    merged = base.merge(saved[cols], on=["question_id", "join_style_id"], how="left")
    if "trace_len" not in merged.columns:
        merged["trace_len"] = -1.0
    if merged["oracle_style_name"].isna().any():
        n_miss = int(merged["oracle_style_name"].isna().sum())
        print(f"  warn: {n_miss} rows missing oracle meta; filling from style_id")
        merged["oracle_style_name"] = merged["oracle_style_name"].fillna(
            merged["join_style_id"].astype(str)
        )
        merged["oracle_style_id"] = (
            merged["oracle_style_id"].fillna(merged["join_style_id"]).astype(np.int32)
        )
    return merged


def load_val_meta(gmm_dir: Path, emb_val) -> pd.DataFrame:
    saved = pd.read_parquet(gmm_dir / "validation.parquet").copy()
    saved["question_id"] = saved["question_id"].astype(str)
    # val parquet uses same schema as train
    if "oracle_style_id" in saved.columns:
        saved["join_style_id"] = saved["oracle_style_id"].astype(np.int32)
    else:
        saved["join_style_id"] = saved["style_id"].astype(np.int32)
    base = pd.DataFrame(
        {
            "question_id": np.asarray(emb_val["question_id"]).astype(str),
            "join_style_id": np.asarray(emb_val["style_id"]).astype(np.int32),
        }
    )
    merged = base.merge(
        saved[
            [
                c
                for c in [
                    "question_id",
                    "join_style_id",
                    "oracle_style_name",
                    "oracle_style_id",
                    "trace_len",
                ]
                if c in saved.columns or c in ("question_id", "join_style_id")
            ]
        ],
        on=["question_id", "join_style_id"],
        how="left",
        validate="one_to_one",
    )
    if "oracle_style_name" not in merged.columns or merged["oracle_style_name"].isna().any():
        # Fall back to embedding style_id only
        merged["oracle_style_name"] = merged["join_style_id"].astype(str)
        merged["oracle_style_id"] = merged["join_style_id"]
        if "trace_len" not in merged.columns:
            merged["trace_len"] = -1
    return merged


def assignment_df(
    meta: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    name_map: dict[int, str],
) -> pd.DataFrame:
    style_names = [name_map[int(c)] for c in labels]
    style_ids = [int(s.split("_")[1]) - 1 for s in style_names]
    return pd.DataFrame(
        {
            "question_id": meta["question_id"].astype(str).to_numpy(),
            "style_name": meta["oracle_style_name"].astype(str).to_numpy(),
            "style_id": meta["oracle_style_id"].astype(np.int32).to_numpy()
            if "oracle_style_id" in meta.columns
            else meta["join_style_id"].astype(np.int32).to_numpy(),
            "oracle_style_name": meta["oracle_style_name"].astype(str).to_numpy(),
            "trace_len": meta["trace_len"].to_numpy(),
            "cluster_id": labels.astype(np.int32),
            "cluster_style_name": style_names,
            "cluster_style_id": np.asarray(style_ids, dtype=np.int32),
            "k": np.full(len(labels), k, dtype=np.int32),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gmm-dir",
        type=Path,
        default=REPO / "data/scas/cluster_gmm_assignments-qwen3-4b",
        help="Dir with embeddings_train_z.npz (+ optional val) and train.parquet meta",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/scas/cluster_gmm_sweep-qwen3-4b",
    )
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--silhouette-sample", type=int, default=10000)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb = np.load(args.gmm_dir / "embeddings_train_z.npz")
    z = l2_normalize(np.asarray(emb["z_s"], dtype=np.float32))
    meta = load_meta_frame(args.gmm_dir, emb)
    lens = meta["trace_len"].to_numpy().astype(np.float64)
    print(f"Train embeddings: {z.shape}")

    emb_val = None
    meta_val = None
    z_val = None
    val_path = args.gmm_dir / "embeddings_val_z.npz"
    if val_path.is_file() and (args.gmm_dir / "validation.parquet").is_file():
        emb_val = np.load(val_path)
        z_val = l2_normalize(np.asarray(emb_val["z_s"], dtype=np.float32))
        meta_val = load_val_meta(args.gmm_dir, emb_val)
        print(f"Val embeddings:   {z_val.shape}")

    # Compare K=2 to previously saved BIC-selected assignments (optional)
    saved_labels = None
    train_pq = args.gmm_dir / "train.parquet"
    if train_pq.is_file():
        saved_k2 = pd.read_parquet(train_pq)
        if "cluster_id" in saved_k2.columns and "oracle_style_id" in saved_k2.columns:
            saved_k2["question_id"] = saved_k2["question_id"].astype(str)
            saved_k2["join_style_id"] = saved_k2["oracle_style_id"].astype(np.int32)
            saved_joined = meta[["question_id", "join_style_id"]].merge(
                saved_k2[["question_id", "join_style_id", "cluster_id"]],
                on=["question_id", "join_style_id"],
                how="left",
            )
            if saved_joined["cluster_id"].notna().all():
                saved_labels = saved_joined["cluster_id"].to_numpy().astype(np.int32)
            else:
                print("  skip K=2 vs saved ARI (incomplete cluster_id join)")
        else:
            print("  skip K=2 vs saved ARI (no prior cluster_id in train.parquet)")
    else:
        print("  no prior train.parquet — fitting GMM from embeddings only")

    sweep = []
    rng = np.random.RandomState(args.seed)
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n=== Fit GMM K={k} on full train ===", flush=True)
        gmm = fit_gmm(z, k, args.seed)
        labels = gmm.predict(np.asarray(z, dtype=np.float64)).astype(np.int32)
        name_map = cluster_to_style_names(labels, lens, k)
        assign = assignment_df(meta, labels, k, name_map)
        out_p = args.output_dir / f"assignments_gmm_k{k:02d}.parquet"
        assign.to_parquet(out_p, index=False)
        model_p = args.output_dir / f"gmm_k{k:02d}.joblib"
        joblib.dump(gmm, model_p)
        print(f"Wrote {out_p} and {model_p}")

        if z_val is not None and meta_val is not None:
            val_labels = gmm.predict(np.asarray(z_val, dtype=np.float64)).astype(np.int32)
            # reuse train name_map so style_i meaning matches train
            val_assign = assignment_df(meta_val, val_labels, k, name_map)
            val_out = args.output_dir / f"assignments_gmm_val_k{k:02d}.parquet"
            val_assign.to_parquet(val_out, index=False)
            print(f"Wrote {val_out}")
        else:
            val_labels = None

        # metrics
        sil = None
        if args.silhouette_sample > 0 and len(z) > k:
            n_s = min(args.silhouette_sample, len(z))
            idx = rng.choice(len(z), size=n_s, replace=False)
            try:
                sil = float(silhouette_score(z[idx], labels[idx], metric="euclidean"))
            except Exception as exc:  # noqa: BLE001
                sil = None
                print(f"  silhouette failed: {exc}")

        mean_len = {}
        for c in range(k):
            mask = labels == c
            mean_len[str(c)] = float(np.mean(lens[mask])) if np.any(mask) else None
        counts = {str(c): int((labels == c).sum()) for c in range(k)}
        n_empty = sum(1 for v in counts.values() if v == 0)
        if n_empty:
            print(f"  warning: {n_empty} empty component(s) at K={k}")
        row = {
            "k": k,
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "n_empty_components": n_empty,
            "train_bic": float(gmm.bic(np.asarray(z, dtype=np.float64))),
            "train_log_likelihood_mean": float(gmm.score(np.asarray(z, dtype=np.float64))),
            "silhouette_subsample": sil,
            "cluster_counts": counts,
            "mean_trace_len_by_cluster": mean_len,
            "cluster_to_style": {str(c): name_map[c] for c in range(k)},
            "majority_teacher": {},
        }
        for c in range(k):
            sub = assign.loc[assign["cluster_id"] == c, "oracle_style_name"]
            if len(sub):
                maj = Counter(sub).most_common(1)[0]
                row["majority_teacher"][str(c)] = {
                    "name": str(maj[0]),
                    "count": int(maj[1]),
                    "frac": float(maj[1] / len(sub)),
                }
        if k == 2 and saved_labels is not None:
            row["ari_vs_saved_bic_k2"] = float(adjusted_rand_score(saved_labels, labels))
            print(f"  ARI vs saved BIC K=2 assignments: {row['ari_vs_saved_bic_k2']:.4f}")
        if val_labels is not None:
            row["val_cluster_counts"] = {
                str(c): int((val_labels == c).sum()) for c in range(k)
            }
        sweep.append(row)
        print(
            f"  bic={row['train_bic']:.1f}  sil={sil}  sizes={counts}  "
            f"styles={row['cluster_to_style']}"
        )

    payload = {
        "method": "gmm_z_full_train_sweep",
        "gmm_dir": str(args.gmm_dir),
        "output_dir": str(args.output_dir),
        "n_train": int(len(z)),
        "n_val": int(len(z_val)) if z_val is not None else 0,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "seed": args.seed,
        "gmm_hyperparams": {
            "covariance_type": "diag",
            "n_init": 5,
            "max_iter": 300,
            "reg_covar": 1e-3,
        },
        "note": (
            "Each K fitted on full L2-normalized train z_s (not BIC-selected only). "
            "style_1 = shortest mean trace_len. "
            "Original pipeline selected K=2 by val BIC; see ari_vs_saved_bic_k2."
        ),
        "sweep": sweep,
        "best_train_bic": min(sweep, key=lambda r: r["train_bic"])["k"],
        "best_silhouette_subsample": max(
            (r for r in sweep if r["silhouette_subsample"] is not None),
            key=lambda r: r["silhouette_subsample"],
            default={"k": None},
        )["k"],
    }
    meta_path = args.output_dir / "gmm_sweep_k2_10.json"
    meta_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
