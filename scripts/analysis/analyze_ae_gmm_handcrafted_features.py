#!/usr/bin/env python3
"""Profile AE-decoder GMM clusters (default K=6) with handcrafted style features.

Reuses the Part-A feature matrix from score_scas_handcrafted_style.py:
  length / pivots / backtracks / formatting (+ density variants).

Reports for each AE GMM source:
  - per-cluster feature means (clusters ordered by mean n_words = style_1 shortest)
  - which features discriminate clusters (eta^2)
  - RandomForest probe: handcrafted feats -> GMM cluster (full vs density)
  - ARI / NMI vs handcrafted k-means at the same K
  - length confounding: MI(cluster; lengthQ) and residual eta^2 within length quartile
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DENSITY_FEATURES = [
    "avg_word_len",
    "avg_sent_len_words",
    "dens_realization",
    "dens_verification",
    "dens_exploration",
    "dens_integration",
    "dens_piv_total",
    "dens_backtrack",
    "dens_equals",
    "dens_boxed",
    "dens_question",
    "dens_steps",
    "lines_per_100w",
]

# Compact set for human-readable cluster profiles
PROFILE_FEATURES = [
    "n_words",
    "n_lines",
    "avg_sent_len_words",
    "piv_total",
    "n_backtrack",
    "n_steps",
    "n_equals",
    "n_boxed",
    "n_question",
    "dens_piv_total",
    "dens_backtrack",
    "dens_verification",
    "dens_realization",
    "dens_exploration",
    "dens_integration",
    "dens_steps",
    "dens_equals",
    "dens_boxed",
    "lines_per_100w",
]


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """ANOVA eta^2 of continuous y explained by discrete groups."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        yg = y[groups == g]
        ss_between += len(yg) * (yg.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    df = pd.crosstab(x, y)
    joint = np.array(df.to_numpy(dtype=np.float64), copy=True)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.where(joint > 0, joint / (px * py), 1.0))
        return float((joint * log_term).sum())


def length_quartiles(lens: np.ndarray) -> np.ndarray:
    qs = np.quantile(lens, [0.25, 0.5, 0.75])
    return np.digitize(lens, qs)


def align_assignments(
    feat_qid: np.ndarray,
    feat_style: np.ndarray,
    assign: pd.DataFrame,
) -> np.ndarray:
    """Return cluster_id aligned to feature-matrix row order."""
    key_to_c = {
        (str(q), str(s)): int(c)
        for q, s, c in zip(
            assign["question_id"].astype(str),
            assign["style_name"].astype(str),
            assign["cluster_id"].astype(int),
        )
    }
    out = np.empty(len(feat_qid), dtype=np.int32)
    missing = 0
    for i, (q, s) in enumerate(zip(feat_qid.astype(str), feat_style.astype(str))):
        c = key_to_c.get((q, s))
        if c is None:
            missing += 1
            out[i] = -1
        else:
            out[i] = c
    if missing:
        raise ValueError(f"{missing}/{len(feat_qid)} feature rows missing from assignments")
    return out


def order_clusters_by_length(labels: np.ndarray, n_words: np.ndarray, k: int) -> dict[int, int]:
    """Map raw cluster_id -> rank 0..k-1 by mean n_words (0 = shortest)."""
    means = []
    for c in range(k):
        m = n_words[labels == c]
        means.append((c, float(m.mean()) if len(m) else float("inf")))
    means.sort(key=lambda t: t[1])
    return {raw: rank for rank, (raw, _) in enumerate(means)}


def probe_cluster(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    maj = float(np.bincount(ytr).max() / len(ytr))
    return {
        "val_accuracy": float((pred == yte).mean()),
        "val_macro_f1": float(f1_score(yte, pred, average="macro")),
        "majority_baseline": maj,
        "feature_importance": sorted(
            [
                {"feature": f"f{i}", "importance": float(v)}
                for i, v in enumerate(clf.feature_importances_)
            ],
            key=lambda d: -d["importance"],
        ),
    }


def analyze_one(
    tag: str,
    labels_raw: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    hand_labels: np.ndarray | None,
    k: int,
) -> dict:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    n_words = X[:, name_to_idx["n_words"]]
    remap = order_clusters_by_length(labels_raw, n_words, k)
    labels = np.array([remap[int(c)] for c in labels_raw], dtype=np.int32)

    counts = {str(c): int((labels == c).sum()) for c in range(k)}
    props = {str(c): counts[str(c)] / len(labels) for c in range(k)}

    # per-cluster means
    profile = {}
    for feat in PROFILE_FEATURES:
        if feat not in name_to_idx:
            continue
        col = X[:, name_to_idx[feat]]
        profile[feat] = {
            f"style_{c+1}": float(col[labels == c].mean()) for c in range(k)
        }
        profile[feat]["eta2"] = eta_squared(col, labels)

    # full eta^2 ranking
    eta_rank = []
    for feat, idx in name_to_idx.items():
        eta_rank.append({"feature": feat, "eta2": eta_squared(X[:, idx], labels)})
    eta_rank.sort(key=lambda d: -d["eta2"])

    # residual eta^2 within length quartile (length-invariant signal)
    lq = length_quartiles(n_words)
    resid_eta = []
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
        resid_eta.append(
            {
                "feature": feat,
                "mean_eta2_within_lenQ": float(np.mean(etas)) if etas else 0.0,
                "n_bins": len(etas),
            }
        )
    resid_eta.sort(key=lambda d: -d["mean_eta2_within_lenQ"])

    # probes
    dens_idx = [name_to_idx[f] for f in DENSITY_FEATURES if f in name_to_idx]
    full_idx = list(range(X.shape[1]))
    # attach real feature names to importances
    probe_full = probe_cluster(X[:, full_idx], labels)
    for item in probe_full["feature_importance"]:
        i = int(item["feature"][1:])
        item["feature"] = feature_names[i]
    probe_dens = probe_cluster(X[:, dens_idx], labels)
    dens_names = [feature_names[i] for i in dens_idx]
    for item in probe_dens["feature_importance"]:
        i = int(item["feature"][1:])
        item["feature"] = dens_names[i]

    # length MI / spearman of cluster rank vs length
    mi_len = mi_discrete(labels, lq)
    rho_len = float(spearmanr(labels, n_words).correlation)

    out = {
        "tag": tag,
        "k": k,
        "n": int(len(labels)),
        "cluster_counts": counts,
        "cluster_props": props,
        "cluster_order_note": "style_i ordered by ascending mean n_words (style_1=shortest)",
        "mean_n_words_by_style": {
            f"style_{c+1}": float(n_words[labels == c].mean()) for c in range(k)
        },
        "mi_cluster_length_quartile": mi_len,
        "spearman_cluster_rank_vs_n_words": rho_len,
        "profile_means": profile,
        "eta2_rank": eta_rank,
        "residual_eta2_within_length_quartile": resid_eta,
        "probe_full": {
            "val_accuracy": probe_full["val_accuracy"],
            "val_macro_f1": probe_full["val_macro_f1"],
            "majority_baseline": probe_full["majority_baseline"],
            "top_features": probe_full["feature_importance"][:12],
        },
        "probe_density": {
            "val_accuracy": probe_dens["val_accuracy"],
            "val_macro_f1": probe_dens["val_macro_f1"],
            "majority_baseline": probe_dens["majority_baseline"],
            "top_features": probe_dens["feature_importance"][:12],
        },
    }

    if hand_labels is not None:
        # remap hand labels by length too for fair ARI? ARI is label-permutation invariant.
        out["vs_handcrafted_kmeans"] = {
            "ari": float(adjusted_rand_score(hand_labels, labels_raw)),
            "nmi": float(normalized_mutual_info_score(hand_labels, labels_raw)),
        }

    return out


def write_summary(results: list[dict], path: Path) -> None:
    lines = ["# AE GMM K=6 × handcrafted style features", ""]
    for r in results:
        lines.append(f"## {r['tag']}")
        lines.append("")
        lines.append(
            f"- n={r['n']}, mean n_words by style: "
            + ", ".join(
                f"{k}={v:.0f}" for k, v in r["mean_n_words_by_style"].items()
            )
        )
        props = ", ".join(f"{k}:{v:.3f}" for k, v in r["cluster_props"].items())
        lines.append(f"- props (style_1..): {props}")
        lines.append(
            f"- MI(cluster; lengthQ)={r['mi_cluster_length_quartile']:.4f}, "
            f"spearman(rank, n_words)={r['spearman_cluster_rank_vs_n_words']:.3f}"
        )
        pf, pd_ = r["probe_full"], r["probe_density"]
        lines.append(
            f"- RF probe FULL: acc={pf['val_accuracy']:.3f} macroF1={pf['val_macro_f1']:.3f} "
            f"(maj={pf['majority_baseline']:.3f})"
        )
        lines.append(
            f"- RF probe DENSITY: acc={pd_['val_accuracy']:.3f} macroF1={pd_['val_macro_f1']:.3f} "
            f"(maj={pd_['majority_baseline']:.3f})"
        )
        if "vs_handcrafted_kmeans" in r:
            v = r["vs_handcrafted_kmeans"]
            lines.append(f"- vs handcrafted k-means K={r['k']}: ARI={v['ari']:.3f} NMI={v['nmi']:.3f}")
        lines.append("")
        lines.append("Top eta^2 features (cluster separation):")
        for row in r["eta2_rank"][:12]:
            lines.append(f"- {row['feature']}: {row['eta2']:.4f}")
        lines.append("")
        lines.append("Top density features with residual eta^2 within length quartile:")
        for row in r["residual_eta2_within_length_quartile"][:10]:
            lines.append(
                f"- {row['feature']}: {row['mean_eta2_within_lenQ']:.4f}"
            )
        lines.append("")
        lines.append("Cluster profile (means):")
        lines.append("")
        # compact table
        feats_show = [
            "n_words",
            "dens_piv_total",
            "dens_backtrack",
            "dens_verification",
            "dens_realization",
            "dens_exploration",
            "dens_integration",
            "dens_steps",
            "dens_equals",
            "dens_boxed",
            "lines_per_100w",
        ]
        header = "| feature | " + " | ".join(f"style_{i}" for i in range(1, r["k"] + 1)) + " | eta2 |"
        sep = "|" + "|".join(["---"] * (r["k"] + 2)) + "|"
        lines.append(header)
        lines.append(sep)
        for feat in feats_show:
            if feat not in r["profile_means"]:
                continue
            row = r["profile_means"][feat]
            cells = [f"{row[f'style_{i}']:.3g}" for i in range(1, r["k"] + 1)]
            lines.append(f"| {feat} | " + " | ".join(cells) + f" | {row['eta2']:.3f} |")
        lines.append("")
        lines.append("RF density top features:")
        for item in pd_["top_features"][:8]:
            lines.append(f"- {item['feature']}: {item['importance']:.4f}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features-npz",
        type=Path,
        default=Path("data/scas/cluster_handcrafted_sweep-qwen3-0p6b/handcrafted_features.npz"),
    )
    p.add_argument(
        "--handcrafted-assign",
        type=Path,
        default=Path(
            "data/scas/cluster_handcrafted_sweep-qwen3-0p6b/assignments_kmeans_k06.parquet"
        ),
    )
    p.add_argument(
        "--gmm-roots",
        type=str,
        default=(
            "covz-qwen3-1p7b:<ARCHIVE_ROOT>/"
            "zscore-variants/covz-qwen3-1p7b/gmm_from_kmeans,"
            "covz-qwen3-4b:<ARCHIVE_ROOT>/"
            "zscore-variants/covz-qwen3-4b/gmm_from_kmeans"
        ),
        help="comma list of tag:dir",
    )
    p.add_argument("--k", type=int, default=6)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get("ARCHIVE_ROOT", "scratch/archive") + "/"
            "zscore-variants/ae_gmm_k6_handcrafted_analysis"
        ),
    )
    args = p.parse_args()

    feat = np.load(args.features_npz, allow_pickle=True)
    X = np.asarray(feat["X"], dtype=np.float64)
    feature_names = [str(x) for x in feat["feature_names"].tolist()]
    qid = feat["question_id"]
    style = feat["style_name"]

    hand_labels = None
    if args.handcrafted_assign.is_file():
        hand = pd.read_parquet(args.handcrafted_assign)
        # keep raw ids; ARI invariant to permutation
        hand_labels = align_assignments(qid, style, hand)

    results = []
    for item in args.gmm_roots.split(","):
        tag, root = item.split(":", 1)
        root = Path(root)
        path = root / f"assignments_gmm_k{args.k:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        assign = pd.read_parquet(path)
        labels = align_assignments(qid, style, assign)
        print(f"=== analyzing {tag} K={args.k} ===")
        r = analyze_one(tag, labels, X, feature_names, hand_labels, args.k)
        results.append(r)
        print(
            f"  probe_full={r['probe_full']['val_accuracy']:.3f} "
            f"probe_dens={r['probe_density']['val_accuracy']:.3f} "
            f"MI_lenQ={r['mi_cluster_length_quartile']:.3f}"
        )
        if "vs_handcrafted_kmeans" in r:
            print(
                f"  vs hand K={args.k}: ARI={r['vs_handcrafted_kmeans']['ari']:.3f} "
                f"NMI={r['vs_handcrafted_kmeans']['nmi']:.3f}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(results, args.out_dir / "SUMMARY.md")
    # also dump a tidy CSV of profiles
    rows = []
    for r in results:
        for feat, stats in r["profile_means"].items():
            row = {"tag": r["tag"], "feature": feat, "eta2": stats["eta2"]}
            for i in range(1, r["k"] + 1):
                row[f"style_{i}"] = stats[f"style_{i}"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "cluster_profiles.csv", index=False)
    print(f"Wrote {args.out_dir}")
    print((args.out_dir / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
