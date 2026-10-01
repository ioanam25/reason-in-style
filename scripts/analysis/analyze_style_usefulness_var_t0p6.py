#!/usr/bin/env python3
"""Functional style differentiation at T=0.6.

Per question x (when every style has >= min_n samples):
  var_z(x) = Var_z[ p(correct | x, z) ]
  mi_x     = I(Z; correct | X=x)   (nats→bits)

Compare Base / vanilla / IS-obs. Base/vanilla use inferred z;
IS-obs uses controlled prefix_style.

  python scripts/analysis/analyze_style_usefulness_var_t0p6.py
"""

from __future__ import annotations
import os

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_style_claims import (  # noqa: E402
    TEACHER_FEATURES_NPZ,
    assign_nearest_style,
    load_teacher_basis,
)

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
RECORDS_ROOT = SROOT / "style_records"
OUT_DIR = SROOT / "style_claims" / "style_usefulness_var_t0p6_is_obs"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6
STYLES = [f"style_{i}" for i in range(1, K + 1)]
MIN_N = {"assigned": 8, "prefix": 16}  # looser for inferred (uneven mass)

CELLS = [
    ("0.6B", "base", "Base",
     "scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-t0p6-eosfix", "assigned"),
    ("0.6B", "vanilla", "vanilla SFT",
     "scas-standard-qwen3-0p6b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "assigned"),
    ("0.6B", "is_obs", "IS-obs",
     "scas-gmm-filter-is-obs-pfx-qwen3-0p6b-base_best-math500-passk-n256x6-t0p6-eosfix", "prefix"),
    ("1.7B", "base", "Base",
     "scas-nosft-qwen3-1p7b-base_pretrained-math500-passk-t0p6-eosfix", "assigned"),
    ("1.7B", "vanilla", "vanilla SFT",
     "scas-standard-qwen3-1p7b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "assigned"),
    ("1.7B", "is_obs", "IS-obs",
     "scas-gmm-filter-is-obs-pfx-qwen3-1p7b-base_best-math500-passk-n256x6-t0p6-eosfix", "prefix"),
]

ARM_ORDER = ["base", "vanilla", "is_obs"]
ARM_COLORS = {"base": "#000000", "vanilla": "#08519c", "is_obs": "#e41a1c"}
ARM_MARKERS = {"base": "o", "vanilla": "*", "is_obs": "X"}
SIZE_X = {"0.6B": 0, "1.7B": 1}


def _entropy_bits_from_p(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def _mi_bits(counts: np.ndarray) -> float:
    """counts shape (K, 2): [n_incorrect, n_correct] per style. I(Z;C) in bits."""
    n = counts.sum()
    if n <= 0:
        return float("nan")
    pz = counts.sum(axis=1) / n
    pc = counts.sum(axis=0) / n
    joint = counts / n
    mi = 0.0
    for i in range(counts.shape[0]):
        for j in range(2):
            if joint[i, j] <= 0 or pz[i] <= 0 or pc[j] <= 0:
                continue
            mi += joint[i, j] * math.log2(joint[i, j] / (pz[i] * pc[j]))
    return float(mi)


def _per_question_metrics(df: pd.DataFrame, min_n: int, require_all_styles: bool) -> dict:
    """If require_all_styles: need every style with >=min_n (IS-obs).
    Else (inferred z): keep styles with >=min_n; require >=2 such styles.
    MI always uses the full empirical (z, correct) table for the question.
    """
    vars_, mis, gaps, n_keep, n_total = [], [], [], 0, 0
    n_styles_used = []
    for _, g in df.groupby("question_id", observed=True):
        n_total += 1
        ps = []
        counts = np.zeros((K, 2), dtype=np.float64)
        present = 0
        for i, s in enumerate(STYLES):
            gs = g[g["z"] == s]
            n = len(gs)
            n_ok = int(gs["correct"].sum()) if n else 0
            counts[i, 1] = n_ok
            counts[i, 0] = n - n_ok
            if n >= min_n:
                present += 1
                ps.append(float(gs["correct"].mean()))
        if require_all_styles and present < K:
            continue
        if (not require_all_styles) and present < 2:
            continue
        n_keep += 1
        n_styles_used.append(present)
        arr = np.asarray(ps, dtype=np.float64)
        vars_.append(float(arr.var(ddof=0)))
        gaps.append(float(arr.max() - arr.mean()))
        mis.append(_mi_bits(counts))
    return {
        "n_questions_total": n_total,
        "n_questions_kept": n_keep,
        "min_n_per_style": min_n,
        "require_all_styles": require_all_styles,
        "mean_n_styles_in_var": float(np.mean(n_styles_used)) if n_styles_used else None,
        "mean_var": float(np.mean(vars_)) if vars_ else None,
        "median_var": float(np.median(vars_)) if vars_ else None,
        "mean_mi_bits": float(np.mean(mis)) if mis else None,
        "median_mi_bits": float(np.median(mis)) if mis else None,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "frac_var_gt_0.0025": float((np.asarray(vars_) > 0.0025).mean()) if vars_ else None,
        "frac_mi_gt_0.02": float((np.asarray(mis) > 0.02).mean()) if mis else None,
        "vars": vars_,
        "mis": mis,
    }


def _load(eval_name: str, z_source: str, basis: dict) -> pd.DataFrame | None:
    p = RECORDS_ROOT / eval_name / "records.parquet"
    if not p.is_file():
        print(f"SKIP {eval_name}", flush=True)
        return None
    df = pd.read_parquet(p).copy()
    if z_source == "prefix":
        df["z"] = df["prefix_style"].astype(str)
    else:
        df["z"] = assign_nearest_style(df, basis)
    return df


def plot_mean_var(rows: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=False)
    metrics = [
        ("mean_var", r"mean $\mathrm{Var}_z[\hat p(c\mid x,z)]$", axes[0]),
        ("mean_mi_bits", r"mean $I(Z;\mathrm{correct}\mid X)$ (bits)", axes[1]),
    ]
    for key, ylab, ax in metrics:
        for arm in ARM_ORDER:
            xs, ys = [], []
            for size, x in SIZE_X.items():
                hit = next((r for r in rows if r["size"] == size and r["arm"] == arm and not r.get("missing")), None)
                if hit is None or hit[key] is None:
                    continue
                xs.append(x)
                ys.append(hit[key])
            if not xs:
                continue
            lab = next(r["label"] for r in rows if r["arm"] == arm)
            ax.plot(
                xs, ys,
                color=ARM_COLORS[arm], marker=ARM_MARKERS[arm], ms=10, lw=2.2,
                mew=1.2, mfc="white", mec=ARM_COLORS[arm], label=lab,
            )
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.4f}" if key == "mean_var" else f"{y:.3f}",
                            xy=(x, y), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=9, color=ARM_COLORS[arm])
        ax.set_xticks(list(SIZE_X.values()))
        ax.set_xticklabels(list(SIZE_X.keys()), fontsize=12)
        ax.set_xlabel("student size", fontsize=12)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=10, loc="best")
    fig.suptitle("Functional style differentiation  ·  MATH-500  ·  T=0.6", fontsize=13, y=1.02)
    fig.tight_layout()
    png = out_dir / "style_usefulness_var_mi_vs_size.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "style_usefulness_var_mi_vs_size.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def plot_var_hist_is_obs(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = {"0.6B": "#08519c", "1.7B": "#e41a1c"}
    for size in ("0.6B", "1.7B"):
        hit = next((r for r in rows if r["size"] == size and r["arm"] == "is_obs" and not r.get("missing")), None)
        if hit is None:
            continue
        v = np.asarray(hit["vars"], dtype=np.float64)
        ax.hist(v, bins=np.linspace(0, 0.08, 40), density=True, histtype="step", lw=2.2,
                color=colors[size],
                label=f"IS-obs {size}  mean={v.mean():.4f}")
        ax.axvline(v.mean(), color=colors[size], ls="--", lw=1.1, alpha=0.85)
    ax.set_xlabel(r"$\mathrm{Var}_z[\hat p(c\mid x,z)]$", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("IS-obs: variation of correctness across styles", fontsize=12)
    fig.tight_layout()
    png = out_dir / "style_var_hist_is_obs.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "style_var_hist_is_obs.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basis = load_teacher_basis(TEACHER_TAG, K, TEACHER_FEATURES_NPZ)
    rows = []
    for size, arm, label, eval_name, z_source in CELLS:
        df = _load(eval_name, z_source, basis)
        if df is None:
            rows.append({"size": size, "arm": arm, "label": label, "eval_dir": eval_name,
                         "z_source": z_source, "missing": True,
                         "mean_var": None, "mean_mi_bits": None})
            continue
        m = _per_question_metrics(df, MIN_N[z_source], require_all_styles=(z_source == "prefix"))
        row = {
            "size": size, "arm": arm, "label": label, "eval_dir": eval_name,
            "z_source": z_source, "missing": False,
            **{k: v for k, v in m.items() if k not in ("vars", "mis")},
            "vars": m["vars"],
            "mis": m["mis"],
        }
        rows.append(row)
        print(
            f"{size:4} {arm:8} kept={m['n_questions_kept']}/{m['n_questions_total']} "
            f"var={m['mean_var']:.5f} mi={m['mean_mi_bits']:.4f} gap={m['mean_gap']:.4f}",
            flush=True,
        )

    # JSON without huge arrays duplicated awkwardly
    slim = []
    for r in rows:
        s = {k: v for k, v in r.items() if k not in ("vars", "mis")}
        slim.append(s)
    summary = {
        "protocol": "T=0.6 MATH-500 all traces",
        "metric": "Var_z[p(c|x,z)] and I(Z;correct|X=x), mean over questions with all styles >= min_n",
        "min_n": MIN_N,
        "z_definition": {
            "base_vanilla": "inferred nearest centroid",
            "is_obs": "prefix_style",
        },
        "rows": slim,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Functional style differentiation (T=0.6)",
        "",
        r"Per question: $\mathrm{Var}_z[\hat p(c\mid x,z)]$ and $I(Z;\mathrm{correct}\mid X=x)$.",
        "",
        "| size | arm | z | kept | mean Var | mean I (bits) | mean gap |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in slim:
        if r.get("missing"):
            lines.append(f"| {r['size']} | {r['label']} | {r['z_source']} | — | — | — | — |")
            continue
        lines.append(
            f"| {r['size']} | {r['label']} | {r['z_source']} | "
            f"{r['n_questions_kept']}/{r['n_questions_total']} | "
            f"{r['mean_var']:.5f} | {r['mean_mi_bits']:.4f} | {r['mean_gap']:.4f} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "summary.md").read_text(), flush=True)

    png1 = plot_mean_var(rows, OUT_DIR)
    png2 = plot_var_hist_is_obs(rows, OUT_DIR)
    print("wrote", png1, flush=True)
    print("wrote", png2, flush=True)


if __name__ == "__main__":
    main()
