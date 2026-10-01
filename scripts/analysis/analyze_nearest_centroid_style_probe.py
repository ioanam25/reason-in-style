#!/usr/bin/env python3
"""Nearest-centroid style probe on an existing MATH-500 eval dump.

Assigns each generation to the nearest teacher GMM centroid in handcrafted
STYLE_SPACE (densities + log length; covz-qwen3-4b-final K=6), then reports
occupancy vs the teacher prior and P(correct | assigned style).

  python scripts/analysis/analyze_nearest_centroid_style_probe.py \
    --eval-dir-name scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-eosfix \
    --arm nosft --student 0.6B-Base \
    --bar-label "no-SFT gens" \
    --title "Qwen3-0.6B-Base no-SFT · MATH-500 · 256 samples · nearest teacher centroid" \
    --out-dir $SROOT/style_claims/nosft_0p6b_base_style_probe
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
from scripts.analysis.extract_style_records import process_dir  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
RECORDS_ROOT = SROOT / "style_records"
TEACHER_TAG = "covz-qwen3-4b-final"
K = 6
STYLE_GLOSS = {
    "style_1": "short textbook",
    "style_2": "short unmarked",
    "style_3": "contest markdown",
    "style_4": "longer markdown",
    "style_5": "default mix",
    "style_6": "long think-block",
}


def _entropy_bits(props: dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in props.values() if p > 0))


def _extract_if_needed(eval_dir: Path, arm: str, student: str, workers: int) -> Path:
    parquet = RECORDS_ROOT / eval_dir.name / "records.parquet"
    if parquet.exists():
        print(f"records exist {parquet}", flush=True)
        return parquet
    meta = {
        "arm": arm,
        "student": student,
        "benchmark": "math500",
        "protocol": "eosfix",
    }
    print(f"extract {eval_dir.name}", flush=True)
    info = process_dir(eval_dir, meta, RECORDS_ROOT, workers, qmod=4, per_cell=4)
    print(f"  rows={info['n_rows']} q={info['n_questions']}", flush=True)
    return parquet


def _style_table(df: pd.DataFrame, styles: list[str], teacher: dict[str, float]) -> list[dict]:
    rows = []
    n = len(df)
    n_ok = int(df["correct"].sum())
    base = float(df["correct"].mean())
    for s in styles:
        g = df[df["assigned"] == s]
        ns = int(len(g))
        nok = int(g["correct"].sum()) if ns else 0
        acc = float(g["correct"].mean()) if ns else float("nan")
        occ = ns / n if n else 0.0
        share_ok = nok / n_ok if n_ok else 0.0
        trunc = (
            float((g["finish_reason"].astype(str) == "length").mean()) if ns else float("nan")
        )
        rows.append(
            {
                "style": s,
                "gloss": STYLE_GLOSS.get(s, s),
                "n": ns,
                "occupancy": occ,
                "teacher_prior": teacher.get(s, 0.0),
                "n_correct": nok,
                "p_correct": acc,
                "lift_vs_base": (acc / base) if base and ns else float("nan"),
                "p_style_given_correct": share_ok,
                "lift_given_correct": (share_ok / occ) if occ else float("nan"),
                "pct_trunc": 100.0 * trunc if ns else float("nan"),
                "mean_n_tokens": float(g["n_tokens"].mean()) if ns else float("nan"),
                "mean_n_words": float(g["n_words"].mean()) if ns and "n_words" in g else float("nan"),
            }
        )
    return rows


def plot_probe(out: dict, out_dir: Path, bar_label: str, title: str) -> tuple[Path, Path]:
    styles = out["styles"]
    labels = [STYLE_GLOSS[s].replace(" ", "\n") for s in styles]
    occ = np.array([r["occupancy"] for r in out["by_style"]], dtype=float)
    prior = np.array([r["teacher_prior"] for r in out["by_style"]], dtype=float)
    acc = np.array([r["p_correct"] for r in out["by_style"]], dtype=float)
    x = np.arange(len(styles))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.bar(x - w / 2, 100 * prior, w, color="#bbbbbb", label="teacher prior", edgecolor="none")
    ax.bar(x + w / 2, 100 * occ, w, color="#4daf4a", label=bar_label, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Occupancy of teacher GMM styles")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    colors = ["#2166ac" if a >= out["base_acc"] else "#b2182b" for a in acc]
    ax.bar(x, 100 * acc, color=colors, edgecolor="none")
    ax.axhline(100 * out["base_acc"], color="#333333", lw=0.9, ls="--", label="overall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("P(correct | assigned style) (%)")
    ax.set_ylim(0, max(15, 100 * float(np.nanmax(acc)) * 1.25))
    ax.set_title("Accuracy by inferred style")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=10, y=1.03)
    fig.tight_layout()
    png = out_dir / f"{out_dir.name}.png"
    pdf = out_dir / f"{out_dir.name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_md(out: dict, heading: str) -> str:
    lines = [
        f"# {heading}",
        "",
        f"Student: {out['student']}, arm `{out['arm']}`. "
        f"Probe: nearest teacher centroid in handcrafted `STYLE_SPACE` "
        f"({out['teacher_tag']} GMM K={out['k']}). No prefix at decode.",
        "",
        f"Eval: `{out['eval_dir']}`",
        "",
        f"Rows: {out['n_rows']:,} · questions: {out['n_questions']} · "
        f"overall P(correct) = {100 * out['base_acc']:.2f}% · "
        f"truncation = {out['pct_trunc']:.1f}%",
        "",
        "## Occupancy vs accuracy",
        "",
        "| style | gloss | share | teacher | P(correct) | lift | P(style\\|correct) | mean tokens | % trunc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out["by_style"]:
        lines.append(
            f"| {r['style']} | {r['gloss']} | {100 * r['occupancy']:.1f}% | "
            f"{100 * r['teacher_prior']:.1f}% | {100 * r['p_correct']:.2f}% | "
            f"{r['lift_vs_base']:.2f}× | {100 * r['p_style_given_correct']:.1f}% | "
            f"{r['mean_n_tokens']:.0f} | {r['pct_trunc']:.1f}% |"
        )
    nt = out["by_style_nontrunc"]
    lines += [
        "",
        "## Same, dropping length-truncated samples",
        "",
        f"Kept {out['n_nontrunc']:,} / {out['n_rows']:,} "
        f"(P(correct) = {100 * out['base_acc_nontrunc']:.2f}%).",
        "",
        "| style | share | P(correct) | lift |",
        "|---|---:|---:|---:|",
    ]
    for r in nt:
        lines.append(
            f"| {r['style']} | {100 * r['occupancy']:.1f}% | "
            f"{100 * r['p_correct']:.2f}% | {r['lift_vs_base']:.2f}× |"
        )
    q = out["within_question"]
    lines += [
        "",
        "## Within-question mix (256 samples, no prefix)",
        "",
        f"Mean unique styles / question: {q['mean_unique']:.2f}. "
        f"Mean entropy: {q['mean_entropy_bits']:.2f} bits. "
        f"Questions with a single inferred style: {100 * q['frac_single']:.1f}%.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-dir-name",
        default="scas-nosft-qwen3-0p6b-base_pretrained-math500-passk-eosfix",
    )
    ap.add_argument("--arm", default="nosft")
    ap.add_argument("--student", default="0.6B-Base")
    ap.add_argument("--bar-label", default="no-SFT gens")
    ap.add_argument(
        "--title",
        default="Qwen3-0.6B-Base no-SFT · MATH-500 · 256 samples · nearest teacher centroid",
    )
    ap.add_argument("--heading", default="Style probe on no-SFT 0.6B-Base MATH-500 generations")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SROOT / "style_claims" / "nosft_0p6b_base_style_probe",
    )
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()

    eval_dir = SROOT / "model-evals" / args.eval_dir_name
    parquet = _extract_if_needed(eval_dir, args.arm, args.student, args.workers)
    print(f"load teacher basis {TEACHER_TAG} K={K}", flush=True)
    basis = load_teacher_basis(TEACHER_TAG, K, TEACHER_FEATURES_NPZ)
    styles = basis["styles"]

    df = pd.read_parquet(parquet)
    print(f"assign {len(df):,} rows", flush=True)
    df = df.copy()
    df["assigned"] = assign_nearest_style(df, basis)

    by_style = _style_table(df, styles, basis["props"])
    nontrunc = df[df["finish_reason"].astype(str) != "length"]
    by_nt = _style_table(nontrunc, styles, basis["props"])

    tmp = df[["question_id", "assigned"]].copy()
    n_unique, q_ent, nq, n_single = [], [], 0, 0
    for _, g in tmp.groupby("question_id", observed=True):
        nq += 1
        counts = g["assigned"].value_counts(normalize=True)
        n_u = int(g["assigned"].nunique())
        n_unique.append(n_u)
        q_ent.append(_entropy_bits(counts.to_dict()))
        if n_u <= 1:
            n_single += 1

    occ_h = _entropy_bits({r["style"]: r["occupancy"] for r in by_style})
    out = {
        "eval_dir": eval_dir.name,
        "arm": args.arm,
        "student": args.student,
        "teacher_tag": TEACHER_TAG,
        "k": K,
        "style_space": STYLE_SPACE,
        "styles": styles,
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "base_acc": float(df["correct"].mean()),
        "pct_trunc": 100.0 * float((df["finish_reason"].astype(str) == "length").mean()),
        "n_nontrunc": int(len(nontrunc)),
        "base_acc_nontrunc": float(nontrunc["correct"].mean()) if len(nontrunc) else float("nan"),
        "occupancy_entropy_bits": occ_h,
        "by_style": by_style,
        "by_style_nontrunc": by_nt,
        "within_question": {
            "mean_unique": float(np.mean(n_unique)) if n_unique else None,
            "mean_entropy_bits": float(np.mean(q_ent)) if q_ent else None,
            "frac_single": float(n_single / nq) if nq else None,
        },
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir.name
    js = args.out_dir / f"{stem}.json"
    js.write_text(json.dumps(out, indent=2) + "\n")
    md = render_md(out, args.heading)
    (args.out_dir / f"{stem}.md").write_text(md)
    png, _ = plot_probe(out, args.out_dir, args.bar_label, args.title)
    print(f"wrote {js}", flush=True)
    print(f"wrote {png}", flush=True)
    print(md, flush=True)


if __name__ == "__main__":
    main()
