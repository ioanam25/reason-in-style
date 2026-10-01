#!/usr/bin/env python3
"""No-SFT style occupancy: where HF Base / Instruct / Thinking land in the
teacher GMM, with no prefix SFT and no style cue.

Reads compact records from extract_style_records.py for the three 4B no-SFT
MATH-500 dumps, assigns each generation to the nearest 4B-final GMM centroid
in the same handcrafted space used by intended-vs-realized, and reports:

  - mix over style_1..style_6 vs the teacher prior
  - entropy / max-cluster share (collapse of the init itself)
  - per-question unique-style count and entropy (does 256 samples mix basins?)
  - pairwise total variation between inits
  - mean surface profile (length, asides, latex)

Usage (on the training cluster, after extract):
  python scripts/analysis/analyze_nosft_occupancy.py
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
    STYLE_SPACE,
    TEACHER_FEATURES_NPZ,
    assign_nearest_style,
    load_teacher_basis,
)
from scripts.handcrafted_style_features import PROFILE_FEATURES  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
RECORDS_ROOT = SROOT / "style_records"
DEFAULT_OUT = SROOT / "style_claims" / "nosft_occupancy"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6

STUDENTS = [
    ("4B-Base", "scas-nosft-qwen3-4b-base_pretrained-math500-passk-eosfix"),
    ("4B-Instruct", "scas-nosft-qwen3-4b-instruct_pretrained-math500-passk-eosfix"),
    ("4B-Thinking", "scas-nosft-qwen3-4b-thinking_pretrained-math500-passk-eosfix"),
]

# Fallback if the dir was stored without the -eosfix suffix.
STUDENT_GLOBS = {
    "4B-Base": "scas-nosft-qwen3-4b-base_pretrained-math500-passk*",
    "4B-Instruct": "scas-nosft-qwen3-4b-instruct_pretrained-math500-passk*",
    "4B-Thinking": "scas-nosft-qwen3-4b-thinking_pretrained-math500-passk*",
}


def _entropy_bits(props: dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in props.values() if p > 0))


def _tv(p: dict[str, float], q: dict[str, float], styles: list[str]) -> float:
    return 0.5 * float(sum(abs(p.get(s, 0.0) - q.get(s, 0.0)) for s in styles))


def find_records(student: str, preferred: str, records_root: Path) -> Path:
    hit = records_root / preferred / "records.parquet"
    if hit.exists():
        return hit
    matches = sorted(records_root.glob(STUDENT_GLOBS[student] + "/records.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"no records.parquet for {student} under {records_root} "
            f"(extract the no-SFT MATH dumps first)"
        )
    return matches[-1]


def occupancy_one(df: pd.DataFrame, basis: dict) -> dict:
    styles = basis["styles"]
    realized = assign_nearest_style(df, basis)
    props = {s: float((realized == s).mean()) for s in styles}
    max_s = max(props, key=props.get)

    tmp = pd.DataFrame({"question_id": df["question_id"].to_numpy(), "style": realized})
    n_unique = []
    q_ent = []
    q_max = []
    singleton = 0
    nq = 0
    for _, g in tmp.groupby("question_id", observed=True):
        nq += 1
        counts = g["style"].value_counts(normalize=True)
        n_u = int(g["style"].nunique())
        n_unique.append(n_u)
        q_ent.append(_entropy_bits(counts.to_dict()))
        q_max.append(float(counts.max()))
        if n_u <= 1:
            singleton += 1

    prof = {}
    for f in PROFILE_FEATURES:
        if f in df.columns:
            prof[f] = float(df[f].mean())

    return {
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "occupancy": props,
        "teacher_prior": basis["props"],
        "entropy_bits": _entropy_bits(props),
        "max_entropy_bits": float(math.log2(len(styles))),
        "max_cluster": max_s,
        "max_cluster_share": props[max_s],
        "tv_vs_teacher": _tv(props, basis["props"], styles),
        "mean_unique_styles_per_question": float(np.mean(n_unique)) if n_unique else None,
        "mean_question_entropy_bits": float(np.mean(q_ent)) if q_ent else None,
        "mean_question_max_share": float(np.mean(q_max)) if q_max else None,
        "frac_questions_single_style": float(singleton / nq) if nq else None,
        "pct_finish_length": round(
            100.0 * float((df["finish_reason"].astype(str) == "length").mean()), 2
        )
        if "finish_reason" in df.columns
        else None,
        "mean_n_words": float(df["n_words"].mean()) if "n_words" in df.columns else None,
        "mean_n_tokens": float(df["n_tokens"].mean()) if "n_tokens" in df.columns else None,
        "profile": prof,
        "n_questions_scored": nq,
    }


def plot_occupancy(out: dict, out_dir: Path) -> Path:
    styles = out["styles"]
    students = [s["student"] for s in out["students"]]
    teacher = np.array([out["teacher_prior"][s] for s in styles], dtype=float)
    mat = np.stack(
        [np.array([row["occupancy"][s] for s in styles], dtype=float) for row in out["students"]],
        axis=0,
    )
    x = np.arange(len(styles))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 1.5 * width, teacher, width, label="teacher prior", color="#bbbbbb", edgecolor="none")
    colors = ["#2166ac", "#4daf4a", "#b2182b"]
    for i, (stu, col) in enumerate(zip(students, colors)):
        ax.bar(x + (i - 0.5) * width, mat[i], width, label=stu, color=col, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(styles, fontsize=8)
    ax.set_ylabel("share of generations", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("No-SFT occupancy of teacher GMM styles (MATH-500)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    pdf = out_dir / "fig_nosft_occupancy.pdf"
    png = out_dir / "fig_nosft_occupancy.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return png


def render_md(out: dict) -> str:
    styles = out["styles"]
    lines = [
        "# No-SFT style occupancy (4B Base / Instruct / Thinking)",
        "",
        f"Teacher basis: `{out['teacher_tag']}` GMM K={out['k']} in handcrafted "
        f"`STYLE_SPACE` (densities + log length). MATH-500, unfinetuned HF checkpoints, "
        f"no style prefix.",
        "",
        "## Mix (share of generations assigned to nearest teacher centroid)",
        "",
        "| init | " + " | ".join(styles) + " | entropy | max share | TV vs teacher |",
        "|---|---" + "|---" * len(styles) + "|---|---|---|",
    ]
    t = out["teacher_prior"]
    lines.append(
        "| teacher prior | "
        + " | ".join(f"{100 * t[s]:.1f}%" for s in styles)
        + f" | {_entropy_bits(t):.2f} | {100 * max(t.values()):.1f}% | — |"
    )
    for row in out["students"]:
        p = row["occupancy"]
        lines.append(
            f"| {row['student']} | "
            + " | ".join(f"{100 * p[s]:.1f}%" for s in styles)
            + f" | {row['entropy_bits']:.2f} | {100 * row['max_cluster_share']:.1f}% "
            f"({row['max_cluster']}) | {row['tv_vs_teacher']:.3f} |"
        )
    lines += [
        "",
        "## Within-question diversity (256 samples, no prefix)",
        "",
        "| init | mean unique styles / Q | mean H (bits) | mean max-share | % Q with 1 style | mean words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in out["students"]:
        lines.append(
            f"| {row['student']} | {row['mean_unique_styles_per_question']:.2f} | "
            f"{row['mean_question_entropy_bits']:.2f} | "
            f"{100 * row['mean_question_max_share']:.1f}% | "
            f"{100 * row['frac_questions_single_style']:.1f}% | "
            f"{row['mean_n_words']:.0f} |"
        )
    lines += ["", "## Pairwise total variation", ""]
    pv = out["pairwise_tv"]
    lines.append("| | " + " | ".join(pv.keys()) + " |")
    lines.append("|---|" + "---|" * len(pv))
    for a in pv:
        lines.append("| " + a + " | " + " | ".join(f"{pv[a][b]:.3f}" for b in pv) + " |")
    lines += [
        "",
        "If Thinking's mix is a spike on the long teacher basin and Base is closer to the "
        "teacher prior (or more spread), post-training compressed style occupancy *before* "
        "any prefix SFT. If all three inits look like the teacher mega-cluster, the "
        "bottleneck is the eval domain / decoding, not Instruct vs Thinking.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--features-npz", type=Path, default=TEACHER_FEATURES_NPZ)
    ap.add_argument("--teacher-tag", default=TEACHER_TAG)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()

    print(f"loading teacher basis {args.teacher_tag} K={args.k}", flush=True)
    basis = load_teacher_basis(args.teacher_tag, args.k, args.features_npz)
    styles = basis["styles"]

    students = []
    for student, dirname in STUDENTS:
        parquet = find_records(student, dirname, args.records_root)
        print(f"load {student} {parquet.parent.name}", flush=True)
        df = pd.read_parquet(parquet)
        block = occupancy_one(df, basis)
        block["student"] = student
        block["eval_dir"] = parquet.parent.name
        students.append(block)
        print(
            f"  occ={ {s: round(block['occupancy'][s], 3) for s in styles} } "
            f"H={block['entropy_bits']:.2f} max={block['max_cluster_share']:.3f} "
            f"TVteach={block['tv_vs_teacher']:.3f}",
            flush=True,
        )

    pairwise = {
        a["student"]: {
            b["student"]: _tv(a["occupancy"], b["occupancy"], styles) for b in students
        }
        for a in students
    }

    out = {
        "teacher_tag": args.teacher_tag,
        "k": args.k,
        "styles": styles,
        "teacher_prior": basis["props"],
        "style_space": STYLE_SPACE,
        "students": students,
        "pairwise_tv": pairwise,
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    js = args.out_dir / "nosft_occupancy.json"
    js.write_text(json.dumps(out, indent=2) + "\n")
    md = render_md(out)
    (args.out_dir / "nosft_occupancy.md").write_text(md)
    png = plot_occupancy(out, args.out_dir)
    print(f"wrote {js}", flush=True)
    print(f"wrote {png}", flush=True)
    print(md, flush=True)


if __name__ == "__main__":
    main()
