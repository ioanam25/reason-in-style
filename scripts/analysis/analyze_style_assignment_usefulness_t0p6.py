#!/usr/bin/env python3
"""Useful style→problem assignment at T=0.6 (IS-obs).

From prefix dump:  p_{x,z} = P(correct | x, z)   (controlled styles)
From free dump:    π(z|x)   = inferred style occupancy under question-only decode

Then per question:
  Δ_x      = max_z p_{x,z} - mean_z p_{x,z}          (style variation in usefulness)
  V_π(x)  = sum_z π(z|x) p_{x,z}                   (value of free-gen mix)
  V_unif  = mean_z p_{x,z}
  V_oracle= max_z p_{x,z}
  align   = corr_z( π(z|x), p_{x,z} )              (Pearson over styles)

Compares whether free-gen style mass lands on styles that actually help.

Also scores Base / vanilla free-gen π against the *same* IS-obs p_{x,z}
landscape (cross-model; labeled as such).

  python scripts/analysis/analyze_style_assignment_usefulness_t0p6.py
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
from scripts.analysis.extract_style_records import process_dir  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
EVAL_ROOT = SROOT / "model-evals"
RECORDS_ROOT = SROOT / "style_records"
OUT_DIR = SROOT / "style_claims" / "style_assignment_usefulness_t0p6_is_obs"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6
STYLES = [f"style_{i}" for i in range(1, K + 1)]

# size -> (pfx_eval, free_gen arms)
SIZES = {
    "0.6B": {
        "pfx": "scas-gmm-filter-is-obs-pfx-qwen3-0p6b-base_best-math500-passk-n256x6-t0p6-eosfix",
        "policies": [
            ("is_obs_nopfx", "IS-obs free", "scas-gmm-filter-is-obs-qwen3-0p6b-base_best-math500-passk-t0p6-eosfix",
             {"arm": "is-obs-nopfx", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
            ("base", "Base", "scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-t0p6-eosfix",
             {"arm": "nosft", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
            ("vanilla", "vanilla SFT", "scas-standard-qwen3-0p6b-base_checkpoint-5250-math500-passk-t0p6-eosfix",
             {"arm": "vanilla", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
        ],
    },
    "1.7B": {
        "pfx": "scas-gmm-filter-is-obs-pfx-qwen3-1p7b-base_best-math500-passk-n256x6-t0p6-eosfix",
        "policies": [
            ("is_obs_nopfx", "IS-obs free", "scas-gmm-filter-is-obs-qwen3-1p7b-base_best-math500-passk-t0p6-eosfix",
             {"arm": "is-obs-nopfx", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
            ("base", "Base", "scas-nosft-qwen3-1p7b-base_pretrained-math500-passk-t0p6-eosfix",
             {"arm": "nosft", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
            ("vanilla", "vanilla SFT", "scas-standard-qwen3-1p7b-base_checkpoint-5250-math500-passk-t0p6-eosfix",
             {"arm": "vanilla", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}),
        ],
    },
}


def _ensure_records(eval_name: str, meta: dict, workers: int) -> Path | None:
    eval_dir = EVAL_ROOT / eval_name
    if not (eval_dir / "generations.jsonl").is_file():
        print(f"SKIP missing gens {eval_name}", flush=True)
        return None
    parquet = RECORDS_ROOT / eval_name / "records.parquet"
    if parquet.exists():
        return parquet
    print(f"extract {eval_name}", flush=True)
    process_dir(eval_dir, meta, RECORDS_ROOT, workers, qmod=4, per_cell=4)
    return parquet


def _p_matrix(pfx_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (n_questions, K) matrix of p_{x,z} and question_id list."""
    qids = sorted(pfx_df["question_id"].astype(str).unique())
    # map style -> p
    piv = (
        pfx_df.assign(question_id=pfx_df["question_id"].astype(str))
        .groupby(["question_id", "prefix_style"], observed=True)["correct"]
        .mean()
        .unstack("prefix_style")
        .reindex(index=qids, columns=STYLES)
    )
    return piv.to_numpy(dtype=np.float64), qids


def _pi_matrix(free_df: pd.DataFrame, qids: list[str], basis: dict) -> np.ndarray:
    df = free_df.copy()
    df["z"] = assign_nearest_style(df, basis)
    df["question_id"] = df["question_id"].astype(str)
    # occupancy π(z|x)
    ct = (
        df.groupby(["question_id", "z"], observed=True)
        .size()
        .unstack("z")
        .reindex(index=qids, columns=STYLES)
        .fillna(0.0)
    )
    arr = ct.to_numpy(dtype=np.float64)
    row = arr.sum(axis=1, keepdims=True)
    row = np.clip(row, 1e-12, None)
    return arr / row


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_policy(P: np.ndarray, Pi: np.ndarray) -> dict:
    """P, Pi: (n, K)."""
    V_pi = (Pi * P).sum(axis=1)
    V_unif = P.mean(axis=1)
    V_oracle = P.max(axis=1)
    delta = V_oracle - V_unif
    # fraction of oracle gap captured (0=uniform, 1=oracle); clip
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(delta > 1e-9, (V_pi - V_unif) / delta, np.nan)
    aligns = np.array([_safe_corr(Pi[i], P[i]) for i in range(P.shape[0])], dtype=np.float64)
    return {
        "n": int(P.shape[0]),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "frac_delta_gt_0.05": float((delta > 0.05).mean()),
        "frac_delta_gt_0.10": float((delta > 0.10).mean()),
        "mean_V_pi": float(V_pi.mean()),
        "mean_V_unif": float(V_unif.mean()),
        "mean_V_oracle": float(V_oracle.mean()),
        "mean_lift_vs_unif": float((V_pi - V_unif).mean()),
        "mean_frac_oracle_gap": float(np.nanmean(frac)),
        "mean_align_corr": float(np.nanmean(aligns)),
        "median_align_corr": float(np.nanmedian(aligns)),
        "delta": delta.tolist(),
        "lift": (V_pi - V_unif).tolist(),
        "align": aligns.tolist(),
    }


def plot_delta_and_lift(size_results: dict, out_dir: Path) -> None:
    # Fig 1: Δ_x hist for IS-obs p landscape (same for all policies within size)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"0.6B": "#08519c", "1.7B": "#e41a1c"}
    ax = axes[0]
    bins = np.linspace(0, 0.4, 41)
    for size, pack in size_results.items():
        d = np.asarray(pack["delta_x"], dtype=np.float64)
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2.2, color=colors[size],
                label=f"{size}  mean={d.mean():.3f}")
        ax.axvline(d.mean(), color=colors[size], ls="--", lw=1.1, alpha=0.85)
    ax.set_xlabel(r"$\Delta_x = \max_z p_{x,z} - \mathrm{mean}_z p_{x,z}$", fontsize=11)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title("Style usefulness varies by question (IS-obs prefixes)", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fig 2: mean lift V_π - V_unif by policy
    ax = axes[1]
    # grouped bars
    sizes = [s for s in ("0.6B", "1.7B") if s in size_results]
    policies = ["is_obs_nopfx", "base", "vanilla"]
    labels = {"is_obs_nopfx": "IS-obs free", "base": "Base", "vanilla": "vanilla"}
    pcolors = {"is_obs_nopfx": "#e41a1c", "base": "#000000", "vanilla": "#08519c"}
    x0 = np.arange(len(sizes), dtype=float)
    width = 0.25
    for i, pol in enumerate(policies):
        ys = []
        for size in sizes:
            hit = size_results[size]["policies"].get(pol)
            ys.append(hit["mean_lift_vs_unif"] if hit else float("nan"))
        xpos = x0 + (i - 1) * width
        ax.bar(xpos, [100 * y if y == y else 0 for y in ys], width=width,
               color=pcolors[pol], label=labels[pol], edgecolor="white")
        for xi, y in zip(xpos, ys):
            if y == y:
                ax.text(xi, 100 * y + (0.15 if y >= 0 else -0.35), f"{100*y:+.2f}",
                        ha="center", va="bottom" if y >= 0 else "top", fontsize=8, color=pcolors[pol])
    ax.axhline(0, color="#666666", lw=1.0)
    ax.set_xticks(x0)
    ax.set_xticklabels(sizes, fontsize=12)
    ax.set_ylabel(r"mean $V_\pi - V_{\mathrm{unif}}$  (pp)", fontsize=11)
    ax.set_title(r"Does free-gen $\pi(z\mid x)$ beat uniform over styles?", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "style_assignment_delta_and_lift.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "style_assignment_delta_and_lift.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_value_bars(size_results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(size_results), figsize=(4.2 * max(1, len(size_results)), 4.0), squeeze=False)
    for ax, size in zip(axes[0], size_results.keys()):
        pack = size_results[size]
        # show unif / IS free / oracle; also base/vanilla if present
        names, vals, cols = [], [], []
        names.append("uniform")
        vals.append(100 * pack["mean_V_unif"])
        cols.append("#999999")
        for key, lab, col in [
            ("base", "Base π", "#000000"),
            ("vanilla", "vanilla π", "#08519c"),
            ("is_obs_nopfx", "IS-obs π", "#e41a1c"),
        ]:
            hit = pack["policies"].get(key)
            if not hit:
                continue
            names.append(lab)
            vals.append(100 * hit["mean_V_pi"])
            cols.append(col)
        names.append("oracle")
        vals.append(100 * pack["mean_V_oracle"])
        cols.append("#4daf4a")
        ax.bar(np.arange(len(names)), vals, color=cols, edgecolor="white")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("expected accuracy under mix (%)", fontsize=10)
        ax.set_title(f"{size}  ·  scored on IS-obs " + r"$p_{x,z}$", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "style_assignment_value_vs_oracle.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "style_assignment_value_vs_oracle.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import os
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basis = load_teacher_basis(TEACHER_TAG, K, TEACHER_FEATURES_NPZ)

    size_results = {}
    slim_rows = []

    for size, cfg in SIZES.items():
        pfx_name = cfg["pfx"]
        pfx_meta = {"arm": "is-obs-pfx", "student": f"{size}-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"}
        pfx_par = _ensure_records(pfx_name, pfx_meta, workers)
        if pfx_par is None:
            continue
        pfx_df = pd.read_parquet(pfx_par)
        P, qids = _p_matrix(pfx_df)
        if np.isnan(P).any():
            # drop questions missing a style
            keep = ~np.isnan(P).any(axis=1)
            P = P[keep]
            qids = [q for q, k in zip(qids, keep) if k]
        delta = P.max(axis=1) - P.mean(axis=1)
        pack = {
            "pfx_eval": pfx_name,
            "n_questions": int(P.shape[0]),
            "mean_V_unif": float(P.mean()),
            "mean_V_oracle": float(P.max(axis=1).mean()),
            "mean_delta": float(delta.mean()),
            "delta_x": delta.tolist(),
            "policies": {},
        }
        print(f"\n=== {size}  Δ mean={delta.mean():.4f}  unif={P.mean():.4f}  oracle={P.max(1).mean():.4f}", flush=True)

        for pol_key, pol_label, eval_name, meta in cfg["policies"]:
            par = _ensure_records(eval_name, meta, workers)
            if par is None:
                continue
            free_df = pd.read_parquet(par)
            Pi = _pi_matrix(free_df, qids, basis)
            sc = score_policy(P, Pi)
            pack["policies"][pol_key] = {**{k: v for k, v in sc.items() if k not in ("delta", "lift", "align")},
                                         "label": pol_label, "eval_dir": eval_name}
            # keep arrays only for primary
            if pol_key == "is_obs_nopfx":
                pack["policies"][pol_key]["lift"] = sc["lift"]
                pack["policies"][pol_key]["align"] = sc["align"]
            print(
                f"  {pol_label:12} Vπ={sc['mean_V_pi']:.4f}  lift={sc['mean_lift_vs_unif']:+.4f}  "
                f"fracOracle={sc['mean_frac_oracle_gap']:+.3f}  corr={sc['mean_align_corr']:+.3f}",
                flush=True,
            )
            slim_rows.append({
                "size": size, "policy": pol_key, "label": pol_label,
                "cross_model_p": pol_key != "is_obs_nopfx",
                **{k: v for k, v in sc.items() if k not in ("delta", "lift", "align")},
            })

        size_results[size] = pack

    summary = {
        "protocol": "T=0.6 MATH-500",
        "p_xz": "IS-obs prefix dump (controlled z)",
        "pi_z": "free-gen dump, inferred nearest-centroid z",
        "note": "Base/vanilla π scored on IS-obs p_{x,z} is cross-model; IS-obs free is matched.",
        "rows": slim_rows,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    # also store size-level delta stats without huge arrays in a compact way
    summary["by_size"] = {
        size: {
            "n_questions": pack["n_questions"],
            "mean_delta": pack["mean_delta"],
            "mean_V_unif": pack["mean_V_unif"],
            "mean_V_oracle": pack["mean_V_oracle"],
            "policies": pack["policies"],
        }
        for size, pack in size_results.items()
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Useful style assignment (T=0.6, IS-obs \(p_{x,z}\))",
        "",
        r"$\Delta_x=\max_z p_{x,z}-\mathrm{mean}_z p_{x,z}$.  $V_\pi=\sum_z \pi(z\mid x)\,p_{x,z}$.",
        "",
        "| size | policy | mean Δ | V_unif | V_π | lift | frac oracle gap | align corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size, pack in size_results.items():
        for pol_key, sc in pack["policies"].items():
            lines.append(
                f"| {size} | {sc['label']} | {pack['mean_delta']:.4f} | "
                f"{100*sc['mean_V_unif']:.2f}% | {100*sc['mean_V_pi']:.2f}% | "
                f"{100*sc['mean_lift_vs_unif']:+.2f}pp | {sc['mean_frac_oracle_gap']:+.3f} | "
                f"{sc['mean_align_corr']:+.3f} |"
            )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "summary.md").read_text(), flush=True)

    plot_delta_and_lift(size_results, OUT_DIR)
    plot_value_bars(size_results, OUT_DIR)
    print("wrote figures to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
