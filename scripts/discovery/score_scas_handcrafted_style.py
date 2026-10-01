#!/usr/bin/env python3
"""Part A: interpretable ("handcrafted") style-feature baseline for SCAS traces.

Operationalizes Lippmann & Yang (2504.01738): style = surface structure
(trace length, pivot markers, backtracking, formatting). We compute these
features directly from the trace text, then:
  1. cluster them (k-means K=2..9) and score teacher recovery (ARI/NMI/majority)
     using the SAME metric fns as the AE-z scorer -> directly comparable;
  2. run a supervised probe (RandomForest) predicting the teacher from features,
     with a question-level split -> upper bound on how stylistically separable
     the teachers actually are;
  3. report which features are just length proxies vs. length-invariant, and
     repeat clustering on a length-invariant ("density") feature set.

Outputs a JSON report + k-means assignment parquets (for downstream prefix SFT).

Usage:
  python scripts/discovery/score_scas_handcrafted_style.py \
      --hf-dataset scas_traces_dataset \
      --raw-json configs/scas/scas_modc_dataset.json \
      --output-json eval_results/scas_handcrafted_style-qwen3-0p6b.json \
      --kmeans-out-dir data/scas/cluster_handcrafted_sweep-qwen3-0p6b \
      --min-k 2 --max-k 9
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
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.eval.eval_gemini_styles import load_gemini_with_styles  # noqa: E402
from scripts.discovery.score_scas_z_representation import (  # noqa: E402
    mean_trace_len_by_cluster,
    question_level_split,
    teacher_metrics,
)

# --- pivot / structure lexicons (case-insensitive) -------------------------
PIVOT_PATTERNS = {
    "realization": r"\b(wait|hmm+|oh|oops|actually|hold on|i missed|my mistake|scratch that|never ?mind)\b",
    "verification": r"(let me (check|verify|confirm|make sure|re-?check|double[- ]?check|re-?examine)|to verify|to check|double[- ]?check|checking again|sanity check|let'?s verify)",
    "exploration": r"(what if|another (approach|way|method|idea)|alternativ|on the other hand|let'?s try|let me try|we could also|suppose that)",
    "integration": r"(now i see|this connects|putting (this|it) together|therefore|thus|in conclusion|to summari[sz]e|so the answer|hence|final answer)",
}
BACKTRACK_PATTERN = r"(\bwait\b|\bactually\b|but wait|on second thought|let me reconsider|scratch that|never ?mind|\binstead\b|going back)"
NUMBERED_STEP_PATTERN = r"(?mi)^\s*(?:step\s*\d+|\d+\s*[\.\):])"

_compiled = {k: re.compile(v, re.IGNORECASE) for k, v in PIVOT_PATTERNS.items()}
_bt = re.compile(BACKTRACK_PATTERN, re.IGNORECASE)
_step = re.compile(NUMBERED_STEP_PATTERN)
_sent = re.compile(r"[.!?]+")


def extract_features(text: str) -> dict[str, float]:
    text = text or ""
    n_chars = len(text)
    words = text.split()
    n_words = max(1, len(words))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    sents = [s for s in _sent.split(text) if s.strip()]
    n_sent = max(1, len(sents))
    avg_sent_len_words = n_words / n_sent
    n_newline = text.count("\n")

    piv = {k: len(rx.findall(text)) for k, rx in _compiled.items()}
    piv_total = sum(piv.values())
    n_backtrack = len(_bt.findall(text))
    n_equals = text.count("=")
    n_dollar = text.count("$")
    n_boxed = text.count("\\boxed")
    n_steps = len(_step.findall(text))
    n_question = text.count("?")

    per100 = 100.0 / n_words

    feats = {
        # absolute / verbosity
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "n_lines": float(n_newline + 1),
        "avg_word_len": avg_word_len,
        "avg_sent_len_words": avg_sent_len_words,
        # raw pivot counts
        "piv_realization": float(piv["realization"]),
        "piv_verification": float(piv["verification"]),
        "piv_exploration": float(piv["exploration"]),
        "piv_integration": float(piv["integration"]),
        "piv_total": float(piv_total),
        "n_backtrack": float(n_backtrack),
        # raw structure
        "n_equals": float(n_equals),
        "n_dollar": float(n_dollar),
        "n_boxed": float(n_boxed),
        "n_steps": float(n_steps),
        "n_question": float(n_question),
        # length-invariant densities (per 100 words)
        "dens_realization": piv["realization"] * per100,
        "dens_verification": piv["verification"] * per100,
        "dens_exploration": piv["exploration"] * per100,
        "dens_integration": piv["integration"] * per100,
        "dens_piv_total": piv_total * per100,
        "dens_backtrack": n_backtrack * per100,
        "dens_equals": n_equals * per100,
        "dens_boxed": n_boxed * per100,
        "dens_question": n_question * per100,
        "dens_steps": n_steps * per100,
        "lines_per_100w": (n_newline + 1) * per100,
    }
    return feats


# length-invariant subset (no absolute counts)
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


def kmeans_sweep(X, y_teacher, teacher_names, lengths, min_k, max_k, seed,
                 records=None, styles=None, out_dir: Path | None = None, tag: str = ""):
    sweep, best = [], None
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        m = teacher_metrics(y_teacher, labels)
        sil = None
        if len(set(labels.tolist())) >= 2:
            sil = float(silhouette_score(X, labels, metric="euclidean",
                                         sample_size=min(10000, len(X)), random_state=seed))
        row = {
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette_subsample": sil,
            "cluster_counts": {str(a): int(b) for a, b in sorted(Counter(int(x) for x in labels).items())},
            "mean_trace_len_by_cluster": mean_trace_len_by_cluster(lengths, labels),
            **{kk: vv for kk, vv in m.items() if kk != "cluster_to_majority_teacher_id"},
            "cluster_to_majority_teacher": {
                cid: teacher_names[tid] for cid, tid in m["cluster_to_majority_teacher_id"].items()
            },
        }
        sweep.append(row)
        print(f"  [{tag}] K={k}: ARI={m['ari_vs_teacher']:.4f} NMI={m['nmi_vs_teacher']:.4f} "
              f"maj={m['majority_teacher_accuracy']:.4f} sil={sil}")
        if best is None or m["ari_vs_teacher"] > best["ari_vs_teacher"]:
            best = {"k": k, "ari_vs_teacher": m["ari_vs_teacher"],
                    "nmi_vs_teacher": m["nmi_vs_teacher"],
                    "majority_teacher_accuracy": m["majority_teacher_accuracy"]}
        if out_dir is not None and records is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "question_id": [r["question_id"] for r in records],
                "style_name": styles,
                "style_id": [int(r.get("style_id", -1)) for r in records],
                "trace_len": lengths,
                "cluster_id": labels.astype(np.int32),
                "k": k,
            }).to_parquet(out_dir / f"assignments_kmeans_k{k:02d}.parquet", index=False)
            np.save(out_dir / f"centers_kmeans_k{k:02d}.npy", km.cluster_centers_.astype(np.float32))
    return sweep, best


def teacher_probe(X, y_teacher, qids, seed, feature_names):
    """RandomForest teacher classifier with a question-level split."""
    tr, va = question_level_split(qids, val_fraction=0.2, seed=seed)
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=seed, class_weight="balanced_subsample",
    )
    clf.fit(X[tr], y_teacher[tr])
    pred = clf.predict(X[va])
    acc = float((pred == y_teacher[va]).mean())
    macro_f1 = float(f1_score(y_teacher[va], pred, average="macro"))
    # majority-class baseline on val
    maj = Counter(y_teacher[tr].tolist()).most_common(1)[0][0]
    maj_acc = float((y_teacher[va] == maj).mean())
    imp = sorted(zip(feature_names, clf.feature_importances_.tolist()),
                 key=lambda t: t[1], reverse=True)
    return {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "val_accuracy": acc,
        "val_macro_f1": macro_f1,
        "majority_baseline_accuracy": maj_acc,
        "n_classes": int(len(set(y_teacher.tolist()))),
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in imp],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", type=str, required=True)
    p.add_argument("--raw-json", type=str, default="")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--kmeans-out-dir", type=Path, default=None)
    p.add_argument("--features-npz", type=Path, default=None)
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", type=str, default="handcrafted-style")
    args = p.parse_args()

    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json or None, split="train"))
    print(f"n_records={len(records)}")

    feat_dicts = [extract_features(r["trace"]) for r in records]
    feature_names = list(feat_dicts[0].keys())
    X_full = np.array([[fd[k] for k in feature_names] for fd in feat_dicts], dtype=np.float64)

    lengths = np.array([int(r["trace_len"]) for r in records], dtype=np.int64)
    qids = np.array([str(r["question_id"]) for r in records])
    styles = [str(r.get("style_name") or r.get("oracle_style_name") or "unknown") for r in records]
    le = LabelEncoder()
    y_teacher = le.fit_transform(styles)
    teacher_names = le.classes_.tolist()
    print(f"teachers={len(teacher_names)}: {teacher_names}")

    if args.features_npz is not None:
        args.features_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_npz, X=X_full.astype(np.float32),
                            feature_names=np.array(feature_names),
                            question_id=qids, style_name=np.array(styles))
        print("Saved features ->", args.features_npz)

    # feature vs length (Spearman) and vs teacher separability summary
    feat_vs_length = {}
    for j, name in enumerate(feature_names):
        rho, _ = spearmanr(X_full[:, j], lengths)
        feat_vs_length[name] = float(rho)

    scaler_full = StandardScaler()
    Xs_full = scaler_full.fit_transform(X_full)

    dens_idx = [feature_names.index(f) for f in DENSITY_FEATURES]
    Xs_dens = StandardScaler().fit_transform(X_full[:, dens_idx])

    print("== supervised teacher probe (FULL features) ==")
    probe_full = teacher_probe(Xs_full, y_teacher, qids, args.seed, feature_names)
    print(f"  acc={probe_full['val_accuracy']:.4f} macroF1={probe_full['val_macro_f1']:.4f} "
          f"(majority={probe_full['majority_baseline_accuracy']:.4f}, {probe_full['n_classes']} teachers)")
    print("== supervised teacher probe (DENSITY / length-invariant) ==")
    probe_dens = teacher_probe(Xs_dens, y_teacher, qids, args.seed, DENSITY_FEATURES)
    print(f"  acc={probe_dens['val_accuracy']:.4f} macroF1={probe_dens['val_macro_f1']:.4f}")

    print("== k-means sweep (FULL features) ==")
    sweep_full, best_full = kmeans_sweep(
        Xs_full, y_teacher, teacher_names, lengths, args.min_k, args.max_k, args.seed,
        records=records, styles=styles, out_dir=args.kmeans_out_dir, tag="full")
    print("== k-means sweep (DENSITY / length-invariant) ==")
    dens_out = (args.kmeans_out_dir.parent / (args.kmeans_out_dir.name + "-density")) if args.kmeans_out_dir else None
    sweep_dens, best_dens = kmeans_sweep(
        Xs_dens, y_teacher, teacher_names, lengths, args.min_k, args.max_k, args.seed,
        records=records, styles=styles, out_dir=dens_out, tag="dens")

    out = {
        "label": args.label,
        "hf_dataset": args.hf_dataset,
        "n_points": len(records),
        "n_features_full": len(feature_names),
        "feature_names_full": feature_names,
        "density_feature_names": DENSITY_FEATURES,
        "teacher_names": teacher_names,
        "feature_spearman_vs_length": feat_vs_length,
        "teacher_probe_full": probe_full,
        "teacher_probe_density": probe_dens,
        "kmeans_sweep_full": sweep_full,
        "best_by_ari_full": best_full,
        "kmeans_sweep_density": sweep_dens,
        "best_by_ari_density": best_dens,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print("Wrote", args.output_json)


if __name__ == "__main__":
    main()
