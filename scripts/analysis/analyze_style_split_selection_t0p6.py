#!/usr/bin/env python3
"""Split-sample style selection on Style SFT (IS-obs) MATH-500 dumps.

For each question, split the 256 samples/style into select/eval halves.
  s_x*       = argmax_s p_select[x,s]
  s_global*  = argmax_s mean_x p_select[x,s]
Evaluate on the held-out half; repeat over random splits.

Figure:
  left  — P(s_x*=s_i) from the selection half (mean over splits)
  right — held-out accuracy: uniform / best fixed / per-question

  python scripts/analysis/analyze_style_split_selection_t0p6.py
"""

from __future__ import annotations
import os

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
RECORDS = SROOT / "style_records"
OUT_DIR = SROOT / "style_claims" / "style_split_selection_t0p6_is_obs"

STYLES = [f"style_{i}" for i in range(1, 7)]
STYLE_SHORT = [rf"$s_{i}$" for i in range(1, 7)]
K = 6
N_PER = 256
HALF = N_PER // 2
N_SPLITS = 400
SEED = 0

EVALS = {
    "0.6B": "scas-gmm-filter-is-obs-pfx-qwen3-0p6b-base_best-math500-passk-n256x6-t0p6-eosfix",
    "1.7B": "scas-gmm-filter-is-obs-pfx-qwen3-1p7b-base_best-math500-passk-n256x6-t0p6-eosfix",
    "4B": "scas-gmm-filter-is-obs-pfx-qwen3-4b-base_best-math500-passk-n256x6-t0p6-eosfix",
}
SIZES = list(EVALS.keys())
COLORS = {"0.6B": "#1b9e77", "1.7B": "#d95f02", "4B": "#7570b3"}


def load_cube(eval_name: str) -> tuple[np.ndarray, list[str]]:
    """Return correct cube (n_q, K, 256) and question ids."""
    df = pd.read_parquet(RECORDS / eval_name / "records.parquet")
    df = df.copy()
    df["prefix_style"] = df["prefix_style"].astype(str)
    qids = sorted(df["question_id"].astype(str).unique())
    q_index = {q: i for i, q in enumerate(qids)}
    s_index = {s: i for i, s in enumerate(STYLES)}
    cube = np.full((len(qids), K, N_PER), np.nan, dtype=np.float64)
    # fill in encounter order within each (q,s)
    counts = np.zeros((len(qids), K), dtype=np.int32)
    for q, s, c in zip(
        df["question_id"].astype(str),
        df["prefix_style"],
        df["correct"].astype(np.float64),
    ):
        qi, si = q_index[q], s_index[s]
        j = counts[qi, si]
        if j >= N_PER:
            raise RuntimeError(f"more than {N_PER} samples for {q} {s}")
        cube[qi, si, j] = c
        counts[qi, si] = j + 1
    if not np.isfinite(cube).all():
        raise RuntimeError(f"{eval_name}: incomplete cube {counts.min()}/{counts.max()}")
    if (counts != N_PER).any():
        raise RuntimeError(f"{eval_name}: expected {N_PER} per cell")
    return cube, qids


def run_splits(cube: np.ndarray, n_splits: int, seed: int) -> dict:
    """cube: (n_q, K, 256)."""
    n_q, k, n = cube.shape
    assert k == K and n == N_PER
    rng = np.random.default_rng(seed)

    acc_uniform = np.empty(n_splits)
    acc_fixed = np.empty(n_splits)
    acc_perq = np.empty(n_splits)
    # optional oracle on eval half (upper bound)
    acc_oracle = np.empty(n_splits)
    winner_frac = np.zeros((n_splits, K), dtype=np.float64)
    global_choice = np.zeros(n_splits, dtype=np.int32)

    for t in range(n_splits):
        perm = rng.permutation(n)
        sel_idx, ev_idx = perm[:HALF], perm[HALF:]
        p_sel = cube[:, :, sel_idx].mean(axis=2)  # (n_q, K)
        p_ev = cube[:, :, ev_idx].mean(axis=2)

        s_star = p_sel.argmax(axis=1)  # (n_q,)
        winner_frac[t] = np.bincount(s_star, minlength=K) / n_q

        s_global = int(p_sel.mean(axis=0).argmax())
        global_choice[t] = s_global

        acc_uniform[t] = float(p_ev.mean())
        acc_fixed[t] = float(p_ev[:, s_global].mean())
        acc_perq[t] = float(p_ev[np.arange(n_q), s_star].mean())
        acc_oracle[t] = float(p_ev.max(axis=1).mean())

    def pack(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci95_lo": float(np.quantile(arr, 0.025)),
            "ci95_hi": float(np.quantile(arr, 0.975)),
            "n_splits": int(arr.size),
        }

    lift = acc_perq - acc_fixed
    return {
        "uniform": pack(acc_uniform),
        "best_fixed": pack(acc_fixed),
        "per_question": pack(acc_perq),
        "oracle_eval": pack(acc_oracle),
        "lift_perq_minus_fixed": pack(lift),
        "winner_frac_mean": winner_frac.mean(axis=0).tolist(),
        "winner_frac_std": winner_frac.std(axis=0, ddof=1).tolist(),
        "global_choice_hist": {
            STYLES[i]: float((global_choice == i).mean()) for i in range(K)
        },
        "raw": {
            "uniform": acc_uniform,
            "best_fixed": acc_fixed,
            "per_question": acc_perq,
            "oracle_eval": acc_oracle,
            "winner_frac": winner_frac,
        },
    }


def plot_figure(results: dict[str, dict], out_dir: Path) -> Path:
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # --- left: winner distribution ---
    x = np.arange(K)
    w = 0.26
    for i, size in enumerate(SIZES):
        mu = np.asarray(results[size]["winner_frac_mean"], dtype=np.float64)
        sd = np.asarray(results[size]["winner_frac_std"], dtype=np.float64)
        ax0.bar(
            x + (i - 1) * w, 100.0 * mu, width=w, color=COLORS[size],
            edgecolor="white", linewidth=0.5, label=size,
            yerr=100.0 * 1.96 * sd / np.sqrt(N_SPLITS),
            error_kw={"ecolor": "#333333", "capsize": 2, "lw": 0.9},
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(STYLE_SHORT, fontsize=13)
    ax0.set_ylabel("% of questions selected", fontsize=14)
    ax0.set_xlabel("style", fontsize=14)
    ax0.set_ylim(0, max(42, 100 * max(max(results[s]["winner_frac_mean"]) for s in SIZES) * 1.25))
    ax0.tick_params(axis="y", labelsize=12)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(frameon=False, fontsize=12, title="student", title_fontsize=12)
    ax0.set_title(r"Which style is selected per question", fontsize=14, pad=8)
    ax0.axhline(100.0 / K, color="#888888", ls=":", lw=1.1, zorder=0)

    # --- right: held-out accuracy ---
    methods = [
        ("uniform", "Uniform"),
        ("best_fixed", "Best fixed"),
        ("per_question", "Per-question"),
    ]
    method_colors = {
        "uniform": "#9e9e9e",
        "best_fixed": "#4c78a8",
        "per_question": "#e45756",
    }
    x = np.arange(len(SIZES))
    w = 0.24
    for j, (key, lab) in enumerate(methods):
        means = [100.0 * results[s][key]["mean"] for s in SIZES]
        los = [100.0 * results[s][key]["ci95_lo"] for s in SIZES]
        his = [100.0 * results[s][key]["ci95_hi"] for s in SIZES]
        yerr = np.vstack([np.asarray(means) - los, his - np.asarray(means)])
        bars = ax1.bar(
            x + (j - 1) * w, means, width=w, color=method_colors[key],
            edgecolor="white", linewidth=0.5, label=lab,
            yerr=yerr, error_kw={"ecolor": "#333333", "capsize": 2.5, "lw": 0.9},
        )
        for rect, m in zip(bars, means):
            ax1.text(
                rect.get_x() + rect.get_width() / 2, m + 0.6,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(SIZES, fontsize=13)
    ax1.set_ylabel("Held-out accuracy (%)", fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax1.set_title("Split-sample style selection", fontsize=14, pad=8)
    # headroom for labels
    ymax = max(100.0 * results[s]["per_question"]["ci95_hi"] for s in SIZES)
    ax1.set_ylim(0, min(100, ymax * 1.18 + 2))

    fig.tight_layout(w_pad=2.6)
    png = out_dir / "style_split_selection_t0p6.png"
    pdf = out_dir / "style_split_selection_t0p6.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_sizes = {}
    raw_npz = {}

    for size, eval_name in EVALS.items():
        print(f"load {size} {eval_name}", flush=True)
        cube, qids = load_cube(eval_name)
        print(f"  cube {cube.shape} overall={cube.mean():.4f}", flush=True)
        out = run_splits(cube, N_SPLITS, SEED + hash(size) % 10_000)
        raw = out.pop("raw")
        summary_sizes[size] = {
            "eval_dir": eval_name,
            "n_questions": len(qids),
            "n_per_style": N_PER,
            "half": HALF,
            "n_splits": N_SPLITS,
            **{k: v for k, v in out.items()},
        }
        print(
            f"  uniform={100*out['uniform']['mean']:.2f}  "
            f"fixed={100*out['best_fixed']['mean']:.2f}  "
            f"perq={100*out['per_question']['mean']:.2f}  "
            f"lift={100*out['lift_perq_minus_fixed']['mean']:.2f} "
            f"[{100*out['lift_perq_minus_fixed']['ci95_lo']:.2f},"
            f"{100*out['lift_perq_minus_fixed']['ci95_hi']:.2f}]",
            flush=True,
        )
        raw_npz[size] = raw

    # persist raw arrays
    np.savez_compressed(
        OUT_DIR / "split_selection_raw.npz",
        **{
            f"{size}_{key}": raw_npz[size][key]
            for size in SIZES
            for key in ("uniform", "best_fixed", "per_question", "oracle_eval", "winner_frac")
        },
    )

    summary = {
        "protocol": "Style SFT (IS-obs) · MATH-500 · T=0.6 · 256/style",
        "design": (
            "Per question, randomly split 256 samples/style into 128 select + 128 eval. "
            "s_x* = argmax_s p_select[x,s]; s_global* = argmax_s mean_x p_select[x,s]. "
            "Report held-out means over "
            f"{N_SPLITS} splits with 95% percentile CIs."
        ),
        "methods": {
            "uniform": "mean over styles of p_eval[x,s]",
            "best_fixed": "p_eval[x, s_global*] with s_global* from selection half only",
            "per_question": "p_eval[x, s_x*] with s_x* from selection half only",
        },
        "sizes": summary_sizes,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Split-sample style selection (Style SFT, MATH-500, T=0.6)",
        "",
        f"{N_SPLITS} random 128/128 splits; 95% percentile CIs.",
        "",
        "| size | uniform | best fixed | per-question | lift (perq−fixed) |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        r = summary_sizes[size]
        def fmt(d):
            return f"{100*d['mean']:.2f} [{100*d['ci95_lo']:.2f},{100*d['ci95_hi']:.2f}]"
        lines.append(
            f"| {size} | {fmt(r['uniform'])} | {fmt(r['best_fixed'])} | "
            f"{fmt(r['per_question'])} | {fmt(r['lift_perq_minus_fixed'])} |"
        )
    lines += ["", "## Selection frequencies P(s_x*=s) %", ""]
    header = "| size | " + " | ".join(STYLES) + " |"
    sep = "|---|" + "|".join(["---:"] * K) + "|"
    lines += [header, sep]
    for size in SIZES:
        fr = summary_sizes[size]["winner_frac_mean"]
        lines.append("| " + size + " | " + " | ".join(f"{100*v:.1f}" for v in fr) + " |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "summary.md").read_text(), flush=True)

    png = plot_figure(summary_sizes, OUT_DIR)
    # also copy-friendly name in figures sense
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
