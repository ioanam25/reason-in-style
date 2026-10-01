#!/usr/bin/env python3
"""Style entropy vs student size at T=0.6 (all traces, incl. truncated).

Arms: Base (nosft) / vanilla SFT / IS-obs mixed (prefix+IS, 256x6).
Metrics per arm: mean H(Z|x), mean #distinct styles/q, mean pairwise STYLE_SPACE
cosine similarity, Pass@256.

  python scripts/analysis/analyze_style_entropy_vs_size_t0p6.py
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
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_style_claims import (  # noqa: E402
    STYLE_SPACE,
    TEACHER_FEATURES_NPZ,
    assign_nearest_style,
    load_teacher_basis,
)
from scripts.analysis.extract_style_records import process_dir  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
EVAL_ROOT = SROOT / "model-evals"
RECORDS_ROOT = SROOT / "style_records"
OUT_DIR = SROOT / "style_claims" / "style_entropy_vs_size_t0p6_is_obs"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6
PAIRWISE_CAP = 128  # max traces/question for pairwise cos (entropy uses all)
RNG = np.random.default_rng(0)

# (size_key, arm_key, label, eval_dir_name, meta)
CELLS = [
    (
        "0.6B",
        "base",
        "Base",
        "scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-t0p6-eosfix",
        {"arm": "nosft", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
    (
        "0.6B",
        "vanilla",
        "vanilla SFT",
        "scas-standard-qwen3-0p6b-base_checkpoint-5250-math500-passk-t0p6-eosfix",
        {"arm": "vanilla", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
    (
        "0.6B",
        "is_obs",
        "IS-obs mixed",
        "scas-gmm-filter-is-obs-pfx-qwen3-0p6b-base_best-math500-passk-n256x6-t0p6-eosfix",
        {"arm": "is-obs-pfx", "student": "0.6B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
    (
        "1.7B",
        "base",
        "Base",
        "scas-nosft-qwen3-1p7b-base_pretrained-math500-passk-t0p6-eosfix",
        {"arm": "nosft", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
    (
        "1.7B",
        "vanilla",
        "vanilla SFT",
        "scas-standard-qwen3-1p7b-base_checkpoint-5250-math500-passk-t0p6-eosfix",
        {"arm": "vanilla", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
    (
        "1.7B",
        "is_obs",
        "IS-obs mixed",
        "scas-gmm-filter-is-obs-pfx-qwen3-1p7b-base_best-math500-passk-n256x6-t0p6-eosfix",
        {"arm": "is-obs-pfx", "student": "1.7B-Base", "benchmark": "math500", "protocol": "t0p6-eosfix"},
    ),
]

ARM_ORDER = ["base", "vanilla", "is_obs"]
ARM_COLORS = {"base": "#000000", "vanilla": "#08519c", "is_obs": "#e41a1c"}
ARM_MARKERS = {"base": "o", "vanilla": "*", "is_obs": "X"}
SIZE_X = {"0.6B": 0, "1.7B": 1}


def _entropy_bits(props: dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in props.values() if p > 0))


def _pass256(eval_dir: Path) -> float | None:
    p = eval_dir / "pass_at_k.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    for key in (
        "pass_at_k_cluster_prefix_balanced_formula",
        "mixed",
        "pass_at_k_standard",
        "pass_at_k",
    ):
        m = d.get(key)
        if isinstance(m, dict) and ("256" in m or 256 in m):
            return 100.0 * float(m.get("256", m.get(256)))
    # nested summaries
    for key in ("pass_at_k_cluster_prefix", "by_budget"):
        m = d.get(key)
        if isinstance(m, dict) and ("256" in m or 256 in m):
            return 100.0 * float(m.get("256", m.get(256)))
    return None


def _ensure_records(eval_dir: Path, meta: dict, workers: int) -> Path:
    parquet = RECORDS_ROOT / eval_dir.name / "records.parquet"
    if parquet.exists():
        print(f"records exist {parquet}", flush=True)
        return parquet
    print(f"extract {eval_dir.name}", flush=True)
    info = process_dir(eval_dir, meta, RECORDS_ROOT, workers, qmod=4, per_cell=4)
    print(f"  rows={info['n_rows']} q={info['n_questions']}", flush=True)
    return parquet


def _pairwise_mean_cos(df: pd.DataFrame, cap: int) -> float:
    """Mean within-question pairwise cosine similarity in STYLE_SPACE (higher=more alike)."""
    sims = []
    cols = STYLE_SPACE
    for _, g in df.groupby("question_id", observed=True):
        X = g[cols].to_numpy(dtype=np.float64)
        n = X.shape[0]
        if n < 2:
            continue
        if n > cap:
            idx = RNG.choice(n, size=cap, replace=False)
            X = X[idx]
        # standardize per-question lightly to avoid length domination? use raw STYLE_SPACE
        # (already includes densities + log_n_words). L2-normalize via cosine_similarity.
        S = cosine_similarity(X)
        iu = np.triu_indices(S.shape[0], k=1)
        sims.append(float(S[iu].mean()))
    return float(np.mean(sims)) if sims else float("nan")


def probe_one(size: str, arm: str, label: str, eval_name: str, meta: dict, basis: dict, workers: int) -> dict | None:
    eval_dir = EVAL_ROOT / eval_name
    if not eval_dir.is_dir() or not (eval_dir / "generations.jsonl").is_file():
        print(f"SKIP missing dump {eval_name}", flush=True)
        return None
    parquet = _ensure_records(eval_dir, meta, workers)
    df = pd.read_parquet(parquet)
    # all traces (incl truncated)
    df = df.copy()
    df["log_n_words"] = np.log1p(df["n_words"].to_numpy(dtype=np.float64))
    df["assigned"] = assign_nearest_style(df, basis)

    n_unique, q_ent = [], []
    for _, g in df.groupby("question_id", observed=True):
        counts = g["assigned"].value_counts(normalize=True)
        n_unique.append(int(g["assigned"].nunique()))
        q_ent.append(_entropy_bits(counts.to_dict()))

    print(f"pairwise {eval_name} (cap={PAIRWISE_CAP}/q)", flush=True)
    pair_sim = _pairwise_mean_cos(df, PAIRWISE_CAP)

    occ = df["assigned"].value_counts(normalize=True).to_dict()
    out = {
        "size": size,
        "arm": arm,
        "label": label,
        "eval_dir": eval_name,
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "pct_trunc": 100.0 * float((df["finish_reason"].astype(str) == "length").mean()),
        "mean_entropy_bits": float(np.mean(q_ent)),
        "mean_unique_styles": float(np.mean(n_unique)),
        "mean_pairwise_cos_sim": pair_sim,
        "occupancy": {k: float(v) for k, v in sorted(occ.items())},
        "occupancy_entropy_bits": _entropy_bits({k: float(v) for k, v in occ.items()}),
        "pass_at_256": _pass256(eval_dir),
        "base_acc": float(df["correct"].mean()),
    }
    print(
        f"  {size} {arm}: H={out['mean_entropy_bits']:.3f} "
        f"uniq={out['mean_unique_styles']:.2f} sim={out['mean_pairwise_cos_sim']:.3f} "
        f"@256={out['pass_at_256']}",
        flush=True,
    )
    return out


def plot_entropy(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arm in ARM_ORDER:
        xs, ys = [], []
        for size, x in SIZE_X.items():
            hit = next((r for r in rows if r["size"] == size and r["arm"] == arm), None)
            if hit is None:
                continue
            xs.append(x)
            ys.append(hit["mean_entropy_bits"])
        if not xs:
            continue
        lab = next(r["label"] for r in rows if r["arm"] == arm)
        ax.plot(
            xs,
            ys,
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            ms=10,
            lw=2.2,
            mew=1.2,
            mfc="white",
            mec=ARM_COLORS[arm],
            label=lab,
        )
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=10, color=ARM_COLORS[arm])

    ax.axhline(math.log2(K), color="#888888", ls="--", lw=1.0, label=rf"$\log_2 {K}$")
    ax.set_xticks(list(SIZE_X.values()))
    ax.set_xticklabels(list(SIZE_X.keys()), fontsize=12)
    ax.set_xlabel("student size", fontsize=13)
    ax.set_ylabel(r"mean $H(Z\mid x)$ (bits)", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, math.log2(K) + 0.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    png = out_dir / "style_entropy_vs_size_t0p6_is_obs.png"
    pdf = out_dir / "style_entropy_vs_size_t0p6_is_obs.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    import os

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"load teacher basis {TEACHER_TAG} K={K}", flush=True)
    basis = load_teacher_basis(TEACHER_TAG, K, TEACHER_FEATURES_NPZ)

    rows = []
    for size, arm, label, eval_name, meta in CELLS:
        r = probe_one(size, arm, label, eval_name, meta, basis, workers)
        if r is not None:
            rows.append(r)

    summary = {
        "protocol": "T=0.6 / top_p=0.95 / EOS_FIX / MATH-500",
        "include_truncated": True,
        "is_arm": "is-obs-pfx mixed (256 samples/style)",
        "teacher_tag": TEACHER_TAG,
        "k": K,
        "pairwise_cap_per_question": PAIRWISE_CAP,
        "style_space": STYLE_SPACE,
        "rows": rows,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    js = OUT_DIR / "summary.json"
    js.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {js}", flush=True)

    # markdown table
    lines = [
        "# Style entropy vs size (T=0.6, all traces)",
        "",
        "Arms: Base / vanilla SFT / **IS-obs mixed**. Truncated traces **included**.",
        "",
        "| size | arm | n | %trunc | mean H(Z|x) | mean #styles/q | mean pairwise cos | Pass@256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p256 = "—" if r["pass_at_256"] is None else f"{r['pass_at_256']:.1f}"
        lines.append(
            f"| {r['size']} | {r['label']} | {r['n_rows']:,} | {r['pct_trunc']:.1f} | "
            f"{r['mean_entropy_bits']:.3f} | {r['mean_unique_styles']:.2f} | "
            f"{r['mean_pairwise_cos_sim']:.3f} | {p256} |"
        )
    md = OUT_DIR / "summary.md"
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text(), flush=True)

    png = plot_entropy(rows, OUT_DIR)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
