#!/usr/bin/env python3
"""Workstream 4: why decoder size changes the styles the AE discovers.

All four AE variants embed the *same* 111,834 teacher traces, so their z-spaces can
be compared directly:

  cross-decoder ARI / NMI of the GMM K=6 labels - low agreement is the quantitative
      form of "different decoders discover different styles"
  linear CKA and PCA-Procrustes between the z-spaces - separates "same geometry,
      relabelled clusters" from "genuinely different subspace"
  per-decoder MI(cluster; length quartile) and residual eta^2 of the length-invariant
      features within length quartile

The hypothesis under test: a weak decoder cannot model surface form itself, so the
8-token prefix is pushed onto the crudest globally useful signal (verbosity), while a
larger decoder frees z to carry discourse structure. That predicts MI(cluster; length)
falls and length-residual structure rises with decoder size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_ae_gmm_handcrafted_features import (  # noqa: E402
    eta_squared,
    length_quartiles,
)
from scripts.handcrafted_style_features import DENSITY_FEATURES, PROFILE_FEATURES  # noqa: E402
from scripts.style_registry import AE_DECODER_SIZE, AE_TAGS, ZSCORE_ROOT  # noqa: E402

TEACHER_FEATURES_NPZ = REPO_ROOT / "data/scas/cluster_handcrafted_sweep-qwen3-0p6b/handcrafted_features.npz"
DEFAULT_OUT = ZSCORE_ROOT / "ae_decoder_cluster_comparison"


def load_tag(tag: str, k: int) -> dict:
    """z matrix plus length-ranked GMM labels for one AE variant."""
    zpath = ZSCORE_ROOT / tag / "embeddings_train_z.npz"
    npz = np.load(zpath, allow_pickle=True)
    Z = np.asarray(npz["z_s"], dtype=np.float32)
    qid = npz["question_id"].astype(str)

    apath = ZSCORE_ROOT / tag / "gmm_from_kmeans" / f"assignments_gmm_k{k:02d}.parquet"
    assign = pd.read_parquet(apath)
    if len(assign) != len(Z):
        raise RuntimeError(f"{tag}: {len(assign)} assignments vs {len(Z)} z rows")
    if not np.array_equal(assign["question_id"].astype(str).to_numpy(), qid):
        raise RuntimeError(f"{tag}: assignment row order does not match the z embeddings")

    mean_len = assign.groupby("cluster_id")["trace_len"].mean().sort_values()
    rank = {int(c): r for r, c in enumerate(mean_len.index)}
    labels = np.array([rank[int(c)] for c in assign["cluster_id"]], dtype=np.int32)
    return {
        "tag": tag,
        "decoder_size": AE_DECODER_SIZE.get(tag, tag),
        "Z": Z,
        "labels": labels,
        "question_id": qid,
        "style_name": assign["style_name"].astype(str).to_numpy(),
        "trace_len": assign["trace_len"].to_numpy(),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same rows."""
    X = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    Y = Y.astype(np.float64) - Y.astype(np.float64).mean(axis=0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    denom = np.sqrt((xtx**2).sum()) * np.sqrt((yty**2).sum())
    return float((xty**2).sum() / denom) if denom > 0 else float("nan")


def procrustes_disparity(X: np.ndarray, Y: np.ndarray, dim: int, seed: int, n: int) -> dict:
    """Residual after the best orthogonal map between PCA-reduced z-spaces."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xp = PCA(n_components=dim, random_state=seed).fit_transform(X[idx].astype(np.float64))
    Yp = PCA(n_components=dim, random_state=seed).fit_transform(Y[idx].astype(np.float64))
    Xp -= Xp.mean(axis=0)
    Yp -= Yp.mean(axis=0)
    Xp /= np.linalg.norm(Xp)
    Yp /= np.linalg.norm(Yp)
    R, _ = orthogonal_procrustes(Yp, Xp)
    resid = float(np.linalg.norm(Xp - Yp @ R) ** 2)
    return {"pca_dim": dim, "n": int(len(idx)), "disparity": resid}


def confusion(a: np.ndarray, b: np.ndarray, k: int) -> list[list[float]]:
    """Row-normalized contingency of two label sets on the same rows."""
    m = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        sel = a == i
        if not sel.any():
            continue
        for j in range(k):
            m[i, j] = float((b[sel] == j).mean())
    return m.tolist()


def length_block(labels: np.ndarray, n_words: np.ndarray, X: np.ndarray, name_to_idx: dict) -> dict:
    """How much of a decoder's cluster structure is just verbosity?"""
    lq = length_quartiles(n_words)
    mi = float(mutual_info_score(labels, lq))
    # normalize by the label entropy so decoders with different balance stay comparable
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    h_labels = float(-(p * np.log(p)).sum())
    resid = {}
    for feat in DENSITY_FEATURES:
        if feat not in name_to_idx:
            continue
        col = X[:, name_to_idx[feat]]
        etas = []
        for q in range(4):
            m = lq == q
            if m.sum() < 50 or len(np.unique(labels[m])) < 2:
                continue
            etas.append(eta_squared(col[m], labels[m]))
        resid[feat] = float(np.mean(etas)) if etas else 0.0
    eta_len = eta_squared(np.log1p(n_words.astype(np.float64)), labels)
    top = sorted(resid.items(), key=lambda kv: -kv[1])[:6]
    return {
        "mi_cluster_length_quartile": mi,
        "label_entropy_nats": h_labels,
        "mi_over_label_entropy": float(mi / h_labels) if h_labels > 0 else None,
        "eta2_log_length": float(eta_len),
        "residual_eta2_within_length_quartile": resid,
        "mean_residual_eta2": float(np.mean(list(resid.values()))) if resid else None,
        "top_residual_features": [{"feature": f, "eta2": v} for f, v in top],
        "cluster_props": {
            f"style_{i + 1}": float((labels == i).mean()) for i in range(int(labels.max()) + 1)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",".join(AE_TAGS))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--features-npz", type=Path, default=TEACHER_FEATURES_NPZ)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--procrustes-n", type=int, default=30000)
    ap.add_argument("--silhouette-n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t]
    data = {}
    for tag in tags:
        print(f"loading {tag}", flush=True)
        data[tag] = load_tag(tag, args.k)

    ref = data[tags[0]]
    for tag in tags[1:]:
        if not np.array_equal(data[tag]["question_id"], ref["question_id"]):
            raise RuntimeError(f"{tag} rows are not the same traces as {tags[0]}")
    print(f"all decoders share {len(ref['question_id'])} traces", flush=True)

    # teacher handcrafted features, aligned by (question_id, style_name)
    feat = np.load(args.features_npz, allow_pickle=True)
    X = np.asarray(feat["X"], dtype=np.float64)
    names = [str(x) for x in feat["feature_names"].tolist()]
    name_to_idx = {n: i for i, n in enumerate(names)}
    fkey = {
        (q, s): i
        for i, (q, s) in enumerate(
            zip(feat["question_id"].astype(str), feat["style_name"].astype(str))
        )
    }
    order = np.array([fkey[(q, s)] for q, s in zip(ref["question_id"], ref["style_name"])])
    X = X[order]
    n_words = X[:, name_to_idx["n_words"]]

    rng = np.random.default_rng(args.seed)
    sil_idx = rng.choice(len(X), size=min(args.silhouette_n, len(X)), replace=False)

    per_decoder = {}
    for tag in tags:
        d = data[tag]
        print(f"per-decoder metrics {tag}", flush=True)
        blk = length_block(d["labels"], n_words, X, name_to_idx)
        blk["decoder_size"] = d["decoder_size"]
        blk["silhouette_k6"] = float(
            silhouette_score(d["Z"][sil_idx], d["labels"][sil_idx], metric="euclidean")
        )
        blk["z_dim"] = int(d["Z"].shape[1])
        blk["profile_means"] = {
            f"style_{i + 1}": {
                f: float(X[d["labels"] == i, name_to_idx[f]].mean())
                for f in PROFILE_FEATURES
                if f in name_to_idx
            }
            for i in range(args.k)
        }
        per_decoder[tag] = blk

    pairs = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            print(f"cross-decoder {a} vs {b}", flush=True)
            la, lb = data[a]["labels"], data[b]["labels"]
            pairs[f"{a}|{b}"] = {
                "ari": float(adjusted_rand_score(la, lb)),
                "nmi": float(normalized_mutual_info_score(la, lb)),
                "cka_linear": linear_cka(data[a]["Z"], data[b]["Z"]),
                "procrustes": procrustes_disparity(
                    data[a]["Z"], data[b]["Z"], args.pca_dim, args.seed, args.procrustes_n
                ),
                "label_confusion_row_normalized": confusion(la, lb, args.k),
            }

    out = {
        "k": args.k,
        "tags": tags,
        "n_traces": int(len(ref["question_id"])),
        "per_decoder": per_decoder,
        "pairs": pairs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cross_decoder_zspace.json").write_text(json.dumps(out, indent=2))

    lines = [
        "# Cross-decoder z-space comparison (GMM K=6, identical traces)",
        "",
        f"- traces: {out['n_traces']}",
        "",
        "## Per decoder: how much of the cluster structure is verbosity?",
        "",
        "| decoder | tag | z dim | sil@6 | MI(cluster; lenQ) | MI / H(cluster) | eta2 log-len | mean residual eta2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in tags:
        b = per_decoder[tag]
        lines.append(
            f"| {b['decoder_size']} | `{tag}` | {b['z_dim']} | {b['silhouette_k6']:.3f} | "
            f"{b['mi_cluster_length_quartile']:.4f} | {b['mi_over_label_entropy']:.4f} | "
            f"{b['eta2_log_length']:.4f} | {b['mean_residual_eta2']:.4f} |"
        )
    lines += [
        "",
        "## Cross-decoder agreement and geometry",
        "",
        "| pair | ARI | NMI | linear CKA | Procrustes disparity |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, v in pairs.items():
        a, b = key.split("|")
        lines.append(
            f"| {AE_DECODER_SIZE.get(a, a)} vs {AE_DECODER_SIZE.get(b, b)} | {v['ari']:.3f} | "
            f"{v['nmi']:.3f} | {v['cka_linear']:.3f} | {v['procrustes']['disparity']:.3f} |"
        )
    lines += ["", "## Top length-residual features per decoder", ""]
    for tag in tags:
        b = per_decoder[tag]
        feats = ", ".join(f"{t['feature']} ({t['eta2']:.3f})" for t in b["top_residual_features"])
        lines.append(f"- **{b['decoder_size']}** (`{tag}`): {feats}")
    (args.out_dir / "CROSS_DECODER_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
