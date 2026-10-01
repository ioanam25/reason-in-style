#!/usr/bin/env python3
"""Score SCAS AE z: length predictability + teacher recovery via k-means.

Usage:
  python scripts/discovery/score_scas_z_representation.py \\
      --embeddings-npz data/scas/cluster_gmm_assignments-qwen3-0p6b/embeddings_train_z.npz \\
      --hf-dataset scas_traces_dataset \\
      --raw-json configs/scas/scas_modc_dataset.json \\
      --output-json eval_results/scas_z_score-qwen3-0p6b.json \\
      --kmeans-out-dir data/scas/cluster_kmeans_sweep-qwen3-0p6b \\
      --min-k 2 --max-k 9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.clustering_utils import l2_normalize  # noqa: E402
from scripts.eval.eval_gemini_styles import load_gemini_with_styles  # noqa: E402


def question_level_split(question_ids: np.ndarray, val_fraction: float = 0.1, seed: int = 42):
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(question_ids):
        grouped[str(q)].append(i)
    qids = sorted(grouped.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    n_val = max(1, int(round(len(qids) * val_fraction)))
    val_q = set(qids[:n_val])
    train_idx, val_idx = [], []
    for q, idxs in grouped.items():
        (val_idx if q in val_q else train_idx).extend(idxs)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def length_metrics(z: np.ndarray, lengths: np.ndarray, question_ids: np.ndarray) -> dict:
    # simple correlations on full set
    norms = np.linalg.norm(z, axis=1)
    # also mean of abs dims as a scalar view
    z_mean = z.mean(axis=1)
    pear_n_r, pear_n_p = pearsonr(norms, lengths)
    spear_n_r, spear_n_p = spearmanr(norms, lengths)
    pear_m_r, pear_m_p = pearsonr(z_mean, lengths)
    spear_m_r, spear_m_p = spearmanr(z_mean, lengths)

    train_idx, val_idx = question_level_split(question_ids)
    z_tr, z_va = z[train_idx], z[val_idx]
    y_tr, y_va = lengths[train_idx].astype(np.float64), lengths[val_idx].astype(np.float64)

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipe,
        {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
        n_jobs=4,
    )
    search.fit(z_tr, y_tr)
    pred = search.predict(z_va)
    r, p = pearsonr(y_va, pred)
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "corr_z_norm_vs_length": {
            "pearson_r": float(pear_n_r),
            "pearson_p": float(pear_n_p),
            "spearman_r": float(spear_n_r),
            "spearman_p": float(spear_n_p),
        },
        "corr_z_mean_vs_length": {
            "pearson_r": float(pear_m_r),
            "pearson_p": float(pear_m_p),
            "spearman_r": float(spear_m_r),
            "spearman_p": float(spear_m_p),
        },
        "ridge_probe_length": {
            "best_alpha": float(search.best_params_["ridge__alpha"]),
            "val_r2": float(r2_score(y_va, pred)),
            "val_mae": float(mean_absolute_error(y_va, pred)),
            "val_pearson_r": float(r),
            "val_pearson_p": float(p),
        },
    }


def teacher_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    mapping = {}
    for c in sorted(set(int(x) for x in y_pred)):
        mask = y_pred == c
        counts = Counter(int(t) for t in y_true[mask])
        mapping[int(c)] = counts.most_common(1)[0][0]
    pred_teacher = np.array([mapping[int(c)] for c in y_pred])
    acc = float((pred_teacher == y_true).mean())
    # purity
    purity = 0.0
    for c in set(int(x) for x in y_pred):
        mask = y_pred == c
        purity += Counter(int(t) for t in y_true[mask]).most_common(1)[0][1]
    purity /= max(1, len(y_true))
    return {
        "ari_vs_teacher": ari,
        "nmi_vs_teacher": nmi,
        "majority_teacher_accuracy": acc,
        "cluster_purity": float(purity),
        "cluster_to_majority_teacher_id": {str(k): int(v) for k, v in mapping.items()},
    }


def mean_trace_len_by_cluster(lengths: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {}
    for c in sorted(set(int(x) for x in labels)):
        out[str(c)] = float(np.mean(lengths[labels == c]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-npz", type=Path, required=True)
    p.add_argument("--hf-dataset", type=str, required=True)
    p.add_argument("--raw-json", type=str, default="")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--kmeans-out-dir", type=Path, default=None)
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()

    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json or None, split="train"))
    cached = np.load(args.embeddings_npz, allow_pickle=True)
    z_raw = np.asarray(cached["z_s"], dtype=np.float64)
    qids = np.array([str(q) for q in cached["question_id"]])
    if len(z_raw) != len(records):
        raise ValueError(f"embedding/record mismatch: {len(z_raw)} vs {len(records)}")
    if qids.tolist() != [r["question_id"] for r in records]:
        raise ValueError("question_id order mismatch vs dataset")

    lengths = np.array([int(r["trace_len"]) for r in records], dtype=np.int64)
    styles = [str(r.get("style_name") or r.get("oracle_style_name") or "unknown") for r in records]
    le = LabelEncoder()
    y_teacher = le.fit_transform(styles)
    teacher_names = le.classes_.tolist()

    print(f"n={len(z_raw)} dim={z_raw.shape[1]} teachers={len(teacher_names)}")
    print("teachers:", teacher_names)

    length = length_metrics(z_raw, lengths.astype(np.float64), qids)
    print("length ridge val R2=", length["ridge_probe_length"]["val_r2"])
    print("length pearson |z| vs len=", length["corr_z_norm_vs_length"]["pearson_r"])

    z = l2_normalize(z_raw)
    sweep = []
    best = None
    if args.kmeans_out_dir is not None:
        args.kmeans_out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(args.min_k, args.max_k + 1):
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(z)
        metrics = teacher_metrics(y_teacher, labels)
        sil = None
        if len(set(labels.tolist())) >= 2:
            sil = float(
                silhouette_score(z, labels, metric="euclidean", sample_size=min(10000, len(z)), random_state=args.seed)
            )
        row = {
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette_subsample": sil,
            "cluster_counts": {str(a): int(b) for a, b in sorted(Counter(int(x) for x in labels).items())},
            "mean_trace_len_by_cluster": mean_trace_len_by_cluster(lengths, labels),
            **{kk: vv for kk, vv in metrics.items() if kk != "cluster_to_majority_teacher_id"},
            "cluster_to_majority_teacher": {
                cid: teacher_names[tid]
                for cid, tid in metrics["cluster_to_majority_teacher_id"].items()
            },
        }
        sweep.append(row)
        print(
            f"K={k}: ARI={metrics['ari_vs_teacher']:.4f} NMI={metrics['nmi_vs_teacher']:.4f} "
            f"maj_acc={metrics['majority_teacher_accuracy']:.4f} sil={sil}"
        )
        if best is None or metrics["ari_vs_teacher"] > best["ari_vs_teacher"]:
            best = {
                "k": k,
                "ari_vs_teacher": metrics["ari_vs_teacher"],
                "nmi_vs_teacher": metrics["nmi_vs_teacher"],
                "majority_teacher_accuracy": metrics["majority_teacher_accuracy"],
            }

        if args.kmeans_out_dir is not None:
            df = pd.DataFrame(
                {
                    "question_id": [r["question_id"] for r in records],
                    "style_name": styles,
                    "style_id": [int(r.get("style_id", -1)) for r in records],
                    "trace_len": lengths,
                    "cluster_id": labels.astype(np.int32),
                    "k": k,
                }
            )
            df.to_parquet(args.kmeans_out_dir / f"assignments_kmeans_k{k:02d}.parquet", index=False)
            np.save(args.kmeans_out_dir / f"centers_kmeans_k{k:02d}.npy", km.cluster_centers_.astype(np.float32))

    out = {
        "label": args.label or str(args.embeddings_npz),
        "embeddings_npz": str(args.embeddings_npz),
        "hf_dataset": args.hf_dataset,
        "n_points": len(z_raw),
        "z_dim": int(z_raw.shape[1]),
        "teacher_names": teacher_names,
        "length_metrics": length,
        "kmeans_sweep": sweep,
        "best_by_ari_vs_teacher": best,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print("Wrote", args.output_json)

    if args.kmeans_out_dir is not None:
        (args.kmeans_out_dir / "kmeans_sweep_k2_9.json").write_text(
            json.dumps(
                {
                    "method": "kmeans",
                    "n_points": len(z_raw),
                    "embeddings_npz": str(args.embeddings_npz),
                    "teacher_names": teacher_names,
                    "sweep": sweep,
                    "best_by_ari_vs_teacher": best,
                    "status": "complete",
                },
                indent=2,
            )
        )
        print("Wrote kmeans assignments under", args.kmeans_out_dir)


if __name__ == "__main__":
    main()
