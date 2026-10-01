#!/usr/bin/env python3
"""Style-conditioned correctness at T=0.6 (all traces).

Headline: heatmap of P(correct | z=s) for styles x (size, arm).
  - Base / vanilla: z = nearest teacher centroid (inferred; no prefix at decode)
  - IS-obs: z = prefix_style (controlled)

Second figure: per-question style usefulness gap on IS-obs
  gap(x) = max_z P(correct|x,z) - mean_z P(correct|x,z)

  python scripts/analysis/analyze_style_conditioned_correctness_t0p6.py
"""

from __future__ import annotations
import os

import json
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
OUT_DIR = SROOT / "style_claims" / "style_conditioned_correctness_t0p6_is_obs"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6
STYLES = [f"style_{i}" for i in range(1, K + 1)]
STYLE_SHORT = [rf"$s_{i}$" for i in range(1, K + 1)]

# (col_key, size, arm, label, eval_dir, z_source)
ARM_LABELS = {
    "base": "Base",
    "vanilla": "Vanilla SFT",
    "is_obs": "Style SFT",
}
SIZES = ("0.6B", "1.7B", "4B")
ARMS = ("base", "vanilla", "is_obs")

CELLS = [
    ("0.6B", "base", "Base",
     "scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-t0p6-eosfix", "assigned"),
    ("0.6B", "vanilla", "Vanilla SFT",
     "scas-standard-qwen3-0p6b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "assigned"),
    ("0.6B", "is_obs", "Style SFT",
     "scas-gmm-filter-is-obs-pfx-qwen3-0p6b-base_best-math500-passk-n256x6-t0p6-eosfix", "prefix"),
    ("1.7B", "base", "Base",
     "scas-nosft-qwen3-1p7b-base_pretrained-math500-passk-t0p6-eosfix", "assigned"),
    ("1.7B", "vanilla", "Vanilla SFT",
     "scas-standard-qwen3-1p7b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "assigned"),
    ("1.7B", "is_obs", "Style SFT",
     "scas-gmm-filter-is-obs-pfx-qwen3-1p7b-base_best-math500-passk-n256x6-t0p6-eosfix", "prefix"),
    ("4B", "base", "Base",
     "scas-nosft-qwen3-4b-base_pretrained-math500-passk-t0p6-eosfix", "assigned"),
    ("4B", "vanilla", "Vanilla SFT",
     "scas-standard-qwen3-4b-base_checkpoint-5250-math500-passk-t0p6-eosfix", "assigned"),
    ("4B", "is_obs", "Style SFT",
     "scas-gmm-filter-is-obs-pfx-qwen3-4b-base_best-math500-passk-n256x6-t0p6-eosfix", "prefix"),
]


def _load_with_z(eval_name: str, z_source: str, basis: dict) -> pd.DataFrame | None:
    parquet = RECORDS_ROOT / eval_name / "records.parquet"
    if not parquet.is_file():
        print(f"SKIP no records {eval_name}", flush=True)
        return None
    df = pd.read_parquet(parquet)
    df = df.copy()
    if z_source == "prefix":
        z = df["prefix_style"].astype(str)
        if not z.isin(STYLES).all():
            bad = sorted(set(z) - set(STYLES))
            raise RuntimeError(f"{eval_name}: unexpected prefix styles {bad}")
        df["z"] = z
    elif z_source == "assigned":
        df["z"] = assign_nearest_style(df, basis)
    else:
        raise ValueError(z_source)
    return df


def _p_correct_by_z(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for s in STYLES:
        g = df[df["z"] == s]
        out[s] = float(g["correct"].mean()) if len(g) else float("nan")
    return out


def _occupancy(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    return {s: float((df["z"] == s).sum()) / n if n else float("nan") for s in STYLES}


def _question_gaps(df: pd.DataFrame, min_per_style: int = 16) -> np.ndarray:
    """gap(x) = max_z p(c|x,z) - mean_z p(c|x,z); require min_per_style for every z."""
    gaps = []
    for _, g in df.groupby("question_id", observed=True):
        ps = []
        ok = True
        for s in STYLES:
            gs = g[g["z"] == s]
            if len(gs) < min_per_style:
                ok = False
                break
            ps.append(float(gs["correct"].mean()))
        if not ok:
            continue
        arr = np.asarray(ps, dtype=np.float64)
        gaps.append(float(arr.max() - arr.mean()))
    return np.asarray(gaps, dtype=np.float64)


def plot_heatmap(mats_by_size: dict[str, np.ndarray], out_dir: Path) -> Path:
    """Three side-by-side heatmaps (one per student size)."""
    arm_labels = [ARM_LABELS[a] for a in ARMS]
    all_vals = np.concatenate([m.ravel() for m in mats_by_size.values() if m.size])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmax = max(20.0, float(all_vals.max()) if all_vals.size else 20.0)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#dddddd")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharey=True)
    im = None
    for ax, size in zip(axes, SIZES):
        mat = mats_by_size.get(size)
        if mat is None:
            mat = np.full((K, len(ARMS)), np.nan)
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(arm_labels)))
        ax.set_xticklabels(arm_labels, fontsize=12, rotation=0, ha="center")
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(STYLE_SHORT, fontsize=15)
        ax.set_xlabel(size, fontsize=16, labelpad=10)
        for i in range(K):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = "—" if np.isnan(v) else f"{v:.1f}"
                ax.text(
                    j, i, txt, ha="center", va="center", fontsize=13,
                    color="black" if (np.isnan(v) or v < 0.55 * vmax) else "white",
                )
        ax.tick_params(axis="both", which="both", length=0)
    axes[0].set_ylabel("style", fontsize=16)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label(r"$P(\mathrm{correct}\mid \mathrm{style}=s_i)$ (%)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    fig.subplots_adjust(wspace=0.18, right=0.88)
    png = out_dir / "style_conditioned_correctness_heatmap.png"
    pdf = out_dir / "style_conditioned_correctness_heatmap.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_gaps(gap_by_size: dict[str, np.ndarray], out_dir: Path) -> Path:
    """Two-panel figure: ECDF of Δ_x + share of questions above 5/10pp gaps."""
    colors = {"0.6B": "#1b9e77", "1.7B": "#d95f02", "4B": "#7570b3"}
    size_order = [s for s in SIZES if s in gap_by_size and gap_by_size[s].size]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # --- left: ECDF ---
    for size in size_order:
        gaps = np.sort(gap_by_size[size])
        y = np.arange(1, gaps.size + 1) / gaps.size
        ax0.plot(
            gaps, y, color=colors[size], lw=2.4,
            label=fr"{size}  (mean $\Delta_x$={gaps.mean():.3f})",
        )
    for thr, ls in ((0.05, ":"), (0.10, "--")):
        ax0.axvline(thr, color="#666666", ls=ls, lw=1.15, zorder=0)
        ax0.text(
            thr + 0.004, 0.04, f"{int(100 * thr)} pp",
            fontsize=11, color="#555555", rotation=90, va="bottom",
        )
    ax0.set_xlim(0, 0.35)
    ax0.set_ylim(0, 1.02)
    ax0.set_xlabel(r"$\Delta_x$", fontsize=15)
    ax0.set_ylabel("Fraction of questions", fontsize=14)
    ax0.tick_params(labelsize=12)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(frameon=False, fontsize=11.5, loc="lower right")
    ax0.set_title(r"CDF of per-question gap $\Delta_x$", fontsize=14, pad=8)

    # --- right: threshold rates ---
    x = np.arange(len(size_order))
    w = 0.36
    frac05 = [float((gap_by_size[s] > 0.05).mean()) for s in size_order]
    frac10 = [float((gap_by_size[s] > 0.10).mean()) for s in size_order]
    bars0 = ax1.bar(x - w / 2, [100 * v for v in frac05], width=w, color="#4c78a8",
                    label=r"$\Delta_x > 0.05$", edgecolor="white", linewidth=0.6)
    bars1 = ax1.bar(x + w / 2, [100 * v for v in frac10], width=w, color="#f58518",
                    label=r"$\Delta_x > 0.10$", edgecolor="white", linewidth=0.6)
    for bars in (bars0, bars1):
        for rect in bars:
            h = rect.get_height()
            ax1.text(
                rect.get_x() + rect.get_width() / 2, h + 1.2,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=11,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_order, fontsize=13)
    ax1.set_ylabel("% of MATH-500 questions", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax1.set_title("How often style choice matters", fontsize=14, pad=8)

    fig.tight_layout(w_pad=2.4)
    png = out_dir / "style_usefulness_gap_is_obs.png"
    pdf = out_dir / "style_usefulness_gap_is_obs.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"load teacher basis {TEACHER_TAG}", flush=True)
    basis = load_teacher_basis(TEACHER_TAG, K, TEACHER_FEATURES_NPZ)

    mats_by_size = {sz: np.full((K, len(ARMS)), np.nan) for sz in SIZES}
    cell_rows = []
    gap_by_size: dict[str, np.ndarray] = {}
    col_labels = []
    mat_cols = []

    for size, arm, label, eval_name, z_source in CELLS:
        col_key = f"{size}\n{label}"
        df = _load_with_z(eval_name, z_source, basis)
        if df is None:
            col_labels.append(col_key)
            mat_cols.append([float("nan")] * K)
            cell_rows.append({
                "size": size, "arm": arm, "label": label, "eval_dir": eval_name,
                "z_source": z_source, "missing": True,
            })
            continue
        p_by_z = _p_correct_by_z(df)
        occ = _occupancy(df)
        col_labels.append(col_key)
        vals = [100.0 * p_by_z[s] for s in STYLES]
        mat_cols.append(vals)
        mats_by_size[size][:, ARMS.index(arm)] = np.asarray(vals, dtype=np.float64)
        row = {
            "size": size,
            "arm": arm,
            "label": label,
            "eval_dir": eval_name,
            "z_source": z_source,
            "n_rows": int(len(df)),
            "n_questions": int(df["question_id"].nunique()),
            "p_correct_given_z": {s: p_by_z[s] for s in STYLES},
            "occupancy": occ,
            "overall_acc": float(df["correct"].mean()),
            "missing": False,
        }
        if arm == "is_obs":
            gaps = _question_gaps(df, min_per_style=16)
            row["usefulness_gap"] = {
                "n_questions": int(gaps.size),
                "mean": float(gaps.mean()) if gaps.size else None,
                "median": float(np.median(gaps)) if gaps.size else None,
                "p90": float(np.quantile(gaps, 0.9)) if gaps.size else None,
                "frac_gt_0.05": float((gaps > 0.05).mean()) if gaps.size else None,
                "frac_gt_0.10": float((gaps > 0.10).mean()) if gaps.size else None,
            }
            gap_by_size[size] = gaps
            print(
                f"{size} IS-obs gap: n={gaps.size} mean={gaps.mean():.4f} "
                f"med={np.median(gaps):.4f} frac>0.05={(gaps>0.05).mean():.3f}",
                flush=True,
            )
        cell_rows.append(row)
        print(
            f"{size} {arm}: overall={100*row['overall_acc']:.1f}%  "
            + " ".join(f"{s[-1]}={100*p_by_z[s]:.1f}" for s in STYLES),
            flush=True,
        )

    mat = np.asarray(mat_cols, dtype=np.float64).T  # styles x columns
    summary = {
        "protocol": "T=0.6 / MATH-500 / all traces incl. truncated",
        "z_definition": {
            "base_vanilla": "nearest teacher centroid in STYLE_SPACE (inferred)",
            "is_obs": "prefix_style (controlled)",
        },
        "styles": STYLES,
        "columns": col_labels,
        "heatmap_pct": mat.tolist(),
        "cells": cell_rows,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Style-conditioned correctness (T=0.6)",
        "",
        "Base/vanilla: inferred \(z\). IS-obs: controlled `prefix_style`.",
        "",
        "## Heatmap cells = \(P(\\mathrm{correct}\\mid z=s)\) %",
        "",
    ]
    header = "| style | " + " | ".join(c.replace("\n", " ") for c in col_labels) + " |"
    sep = "|---|" + "|".join(["---:"] * len(col_labels)) + "|"
    lines += [header, sep]
    for i, s in enumerate(STYLES):
        cells = []
        for j in range(mat.shape[1]):
            v = mat[i, j]
            cells.append("—" if np.isnan(v) else f"{v:.1f}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "## IS-obs usefulness gap", ""]
    for size, gaps in gap_by_size.items():
        lines.append(
            f"- **{size}**: n={gaps.size}, mean={gaps.mean():.4f}, median={np.median(gaps):.4f}, "
            f"P(gap>0.05)={100*(gaps>0.05).mean():.1f}%, P(gap>0.10)={100*(gaps>0.10).mean():.1f}%"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "summary.md").read_text(), flush=True)

    png1 = plot_heatmap(mats_by_size, OUT_DIR)
    png2 = plot_gaps(gap_by_size, OUT_DIR)
    print(f"wrote {png1}", flush=True)
    print(f"wrote {png2}", flush=True)


if __name__ == "__main__":
    main()
