#!/usr/bin/env python3
"""Post-hoc style rerank on existing student dumps. No new generation.

For each style_records dump, assign every sample to the nearest teacher-side
GMM centroid in handcrafted space, then ask:

  Pass@k          the dump's usual correctness curve (baseline)
  nearest pick    per question, take the sample closest to a target centroid;
                  report that pick's accuracy
  basin filter    keep only samples assigned to the target style; coverage of
                  questions that have at least one such sample, and accuracy
                  of a random sample from the basin
  correct+style   fraction of questions with a sample that is both correct and
                  in the target basin ("right answer that looks like style j")

Vanilla dumps are the null: the same reranker with no [style_i] prefix.

Usage:
  python scripts/analysis/analyze_style_rerank.py --headline-only
  python scripts/analysis/analyze_style_rerank.py   # all EOS_FIX dumps
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_style_claims import (  # noqa: E402
    STYLE_SPACE,
    load_teacher_basis,
    mean_pass_at_k,
)

RECORDS_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_records"
DEFAULT_OUT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_rerank"
TEACHER_FEATURES_NPZ = REPO_ROOT / "data/scas/cluster_handcrafted_sweep-qwen3-0p6b/handcrafted_features.npz"

BUDGETS = [1, 8, 32, 128, 256]
HEADLINE = {
    ("AE-GMM K=6", "4B-Thinking", "math500"),
    ("AE-GMM K=6", "4B-Base", "math500"),
    ("AE-GMM K=6", "4B-Instruct", "math500"),
    ("vanilla SFT", "4B-Thinking", "math500"),
    ("vanilla SFT", "4B-Base", "math500"),
    ("vanilla SFT", "4B-Instruct", "math500"),
}


def distances_to_centroids(df: pd.DataFrame, basis: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (assigned_style, squared_distance_n_by_k)."""
    d = df.copy()
    d["log_n_words"] = np.log1p(d["n_words"].to_numpy(dtype=np.float64))
    S = d[STYLE_SPACE].to_numpy(dtype=np.float64)
    Sz = (S - basis["mu"]) / basis["sd"]
    d2 = ((Sz[:, None, :] - basis["centroids"][None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    assigned = np.asarray(basis["styles"], dtype=object)[idx]
    return assigned, d2


def _pick_accuracy(correct: np.ndarray, dist: np.ndarray) -> float:
    """Accuracy of the nearest-to-target sample. Ties: first index."""
    if len(correct) == 0:
        return float("nan")
    return float(correct[int(np.argmin(dist))])


def rerank_dump(df: pd.DataFrame, basis: dict, rng: np.random.Generator) -> dict:
    needed = [c for c in STYLE_SPACE if c != "log_n_words"] + ["n_words", "correct", "question_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"records missing columns: {missing}")
    styles = list(basis["styles"])
    assigned, d2 = distances_to_centroids(df, basis)
    work = df.copy()
    work["realized"] = assigned
    for j, s in enumerate(styles):
        work[f"d2_{s}"] = d2[:, j]

    qids = work["question_id"].astype(str)
    out: dict = {
        "n_rows": int(len(work)),
        "n_questions": int(qids.nunique()),
        "pass_at_k": {str(k): mean_pass_at_k(work, k) for k in BUDGETS},
        "realized_mass": {
            s: float((work["realized"] == s).mean()) for s in styles
        },
        "by_target": {},
    }

    grouped = list(work.groupby("question_id", observed=True))
    for j, s in enumerate(styles):
        nearest_acc = []
        basin_cov = []
        basin_random_acc = []
        correct_and_style = []
        basin_sizes = []
        for _, g in grouped:
            dist = g[f"d2_{s}"].to_numpy(dtype=np.float64)
            corr = g["correct"].to_numpy(dtype=bool)
            nearest_acc.append(_pick_accuracy(corr, dist))
            in_basin = g["realized"].to_numpy() == s
            n_b = int(in_basin.sum())
            basin_sizes.append(n_b)
            basin_cov.append(n_b > 0)
            if n_b > 0:
                basin_random_acc.append(float(corr[in_basin][int(rng.integers(0, n_b))]))
            else:
                basin_random_acc.append(float("nan"))
            correct_and_style.append(bool((corr & in_basin).any()))

        # filtered Pass@k over questions that have at least k basin samples
        basin_df = work[work["realized"] == s]
        filtered_pass = {}
        for k in BUDGETS:
            filtered_pass[str(k)] = mean_pass_at_k(basin_df, k)

        out["by_target"][s] = {
            "nearest_pick_accuracy": float(np.nanmean(nearest_acc)),
            "coverage": float(np.mean(basin_cov)),
            "mean_basin_size": float(np.mean(basin_sizes)),
            "basin_random_accuracy": float(np.nanmean(basin_random_acc)),
            "correct_and_in_basin": float(np.mean(correct_and_style)),
            "filtered_pass_at_k": filtered_pass,
        }

    # Any-style "correct and in some requested basin" is not the question;
    # report best-target correct+style as an oracle over style choice.
    out["best_target_correct_and_in_basin"] = max(
        v["correct_and_in_basin"] for v in out["by_target"].values()
    )
    out["mean_nearest_pick_accuracy"] = float(
        np.mean([v["nearest_pick_accuracy"] for v in out["by_target"].values()])
    )
    return out


def render_md(rows: list[dict]) -> str:
    lines = [
        "# Style rerank (post-hoc, no new generation)",
        "",
        "Nearest pick = per question, take the sample closest to the target teacher centroid, then grade it.",
        "Coverage = fraction of questions with at least one sample assigned to that centroid.",
        "Correct ∧ style = fraction of questions with a sample that is both correct and in the target basin.",
        "",
        "| arm | student | bench | Pass@1 | Pass@256 | nearest-pick mean | best correct∧style |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        p = r.get("rerank") or {}
        p1 = (p.get("pass_at_k") or {}).get("1")
        p256 = (p.get("pass_at_k") or {}).get("256")
        def pct(x):
            return "—" if x is None else f"{100 * x:.1f}%"
        lines.append(
            f"| {r.get('arm')} | {r.get('student')} | {r.get('benchmark')} | "
            f"{pct(p1)} | {pct(p256)} | {pct(p.get('mean_nearest_pick_accuracy'))} | "
            f"{pct(p.get('best_target_correct_and_in_basin'))} |"
        )
    lines += ["", "## Per-target (headline dumps)", ""]
    for r in rows:
        if (r.get("arm"), r.get("student"), r.get("benchmark")) not in HEADLINE:
            continue
        lines.append(f"### {r.get('arm')} / {r.get('student')} / {r.get('benchmark')}")
        lines.append("")
        lines.append("| target | nearest pick | coverage | basin size | basin random acc | correct∧style |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        by = (r.get("rerank") or {}).get("by_target") or {}
        for s, v in by.items():
            lines.append(
                f"| {s} | {100 * v['nearest_pick_accuracy']:.1f}% | "
                f"{100 * v['coverage']:.1f}% | {v['mean_basin_size']:.1f} | "
                f"{100 * v['basin_random_accuracy']:.1f}% | "
                f"{100 * v['correct_and_in_basin']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--features-npz", type=Path, default=TEACHER_FEATURES_NPZ)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headline-only", action="store_true")
    ap.add_argument("--default-teacher-tag", default="covz-qwen3-4b-final")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Walk style_records dirs that have meta.json from extract_style_records.
    rec_dirs = sorted(p for p in args.records_root.iterdir() if (p / "records.parquet").exists())
    bases: dict[str, dict] = {}
    rows = []
    for rec_dir in rec_dirs:
        meta_path = rec_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        key = (meta.get("arm"), meta.get("student"), meta.get("benchmark"))
        if args.headline_only and key not in HEADLINE:
            continue
        if meta.get("protocol") not in (None, "eosfix"):
            continue
        tag = meta.get("teacher_tag") or args.default_teacher_tag
        if tag not in bases:
            print(f"loading teacher basis {tag}", flush=True)
            bases[tag] = load_teacher_basis(tag, args.k, args.features_npz)
        print(f"rerank {rec_dir.name} [{key}]", flush=True)
        df = pd.read_parquet(rec_dir / "records.parquet")
        df["question_id"] = df["question_id"].astype(str)
        try:
            rerank = rerank_dump(df, bases[tag], rng)
        except Exception as e:
            print(f"  skip: {e}", flush=True)
            continue
        row = {
            **{k: meta.get(k) for k in ("arm", "student", "scale", "benchmark", "protocol", "eval_dir")},
            "teacher_tag": tag,
            "rerank": rerank,
        }
        rows.append(row)

    (args.out_dir / "style_rerank.json").write_text(json.dumps(rows, indent=2))
    md = render_md(rows)
    (args.out_dir / "STYLE_RERANK.md").write_text(md)
    print(md)
    print(f"n_dumps={len(rows)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
