#!/usr/bin/env python3
"""Per-question solve rates across 0.6B-Base style_i SFT models.

Each arm is a separate prefix-free SFT checkpoint trained only on GMM style_i.
For every MATH-500 / AMC12-2025 problem we take p_success = (#correct)/256 from
the valbest Pass@k dump, then count which style is best.

Usage (cluster):
  python scripts/analysis/analyze_style_i_question_wins_0p6b.py
"""

from __future__ import annotations
import os

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_ROOT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "model-evals"
STYLES = [f"style_{i}" for i in range(1, 7)]
DUMP = {
    "MATH-500": {
        s: f"scas-gmm-filter-style{s[-1]}-qwen3-0p6b-base_best-math500-passk-eosfix"
        for s in STYLES
    },
    "AMC12-2025": {
        s: f"scas-gmm-filter-style{s[-1]}-qwen3-0p6b-base_best-amc12-2025-passk-eosfix"
        for s in STYLES
    },
}


def load_bench(bench: str) -> pd.DataFrame:
    frames = []
    qmeta = {}
    for style, name in DUMP[bench].items():
        rec = json.loads((EVAL_ROOT / name / "pass_at_k.json").read_text())
        pk = rec.get("pass_at_k_standard") or rec["pass_at_k"]
        rows = []
        for item in rec["per_problem"]:
            qid = str(item["question_id"])
            n = int(item.get("n_samples") or 256)
            p = float(item["p_success"])
            qmeta[qid] = item.get("subject")
            rows.append(
                {
                    "question_id": qid,
                    "style": style,
                    "n_samples": n,
                    "n_correct": int(round(p * n)),
                    "p_success": p,
                }
            )
        df = pd.DataFrame(rows)
        df.attrs["pass_at_1"] = float(pk["1"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["subject"] = out["question_id"].map(qmeta)
    out.attrs["benchmark"] = bench
    return out


def summarize(df: pd.DataFrame) -> dict:
    bench = df.attrs["benchmark"]
    wide = df.pivot(index="question_id", columns="style", values="p_success")[STYLES]
    n_corr = df.pivot(index="question_id", columns="style", values="n_correct")[STYLES]
    subject = df.drop_duplicates("question_id").set_index("question_id")["subject"]
    P = wide.to_numpy(dtype=float)
    qids = list(wide.index)
    n_q, k = P.shape
    maxv = P.max(axis=1)
    is_max = P == maxv[:, None]
    n_at_max = is_max.sum(axis=1)
    any_correct = maxv > 0

    unique_wins = {s: 0 for s in STYLES}
    incl_ties = {s: 0 for s in STYLES}
    unique_by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in STYLES})
    for i, qid in enumerate(qids):
        if not any_correct[i]:
            continue
        winners = [STYLES[j] for j in range(k) if is_max[i, j]]
        for s in winners:
            incl_ties[s] += 1
        if len(winners) == 1:
            unique_wins[winners[0]] += 1
            subj = str(subject.loc[qid] or "")
            unique_by_subject[subj][winners[0]] += 1

    n_unsolved = int((~any_correct).sum())
    n_tied = int((any_correct & (n_at_max > 1)).sum())
    n_unique = int((any_correct & (n_at_max == 1)).sum())

    mean_p = {s: float(wide[s].mean()) for s in STYLES}
    n_solved_by = {s: int((wide[s] > 0).sum()) for s in STYLES}

    # disagreement: unique winner and second-best gap
    gaps = []
    for i, qid in enumerate(qids):
        if not (any_correct[i] and n_at_max[i] == 1):
            continue
        row = P[i]
        order = np.argsort(-row)
        best, second = STYLES[int(order[0])], STYLES[int(order[1])]
        gap = float(row[order[0]] - row[order[1]])
        gaps.append(
            {
                "question_id": qid,
                "subject": subject.loc[qid],
                "best": best,
                "second": second,
                "p_best": float(row[order[0]]),
                "p_second": float(row[order[1]]),
                "gap": gap,
                "n_correct": {s: int(n_corr.loc[qid, s]) for s in STYLES},
                "p_success": {s: float(wide.loc[qid, s]) for s in STYLES},
            }
        )
    gaps.sort(key=lambda r: -r["gap"])

    # pairwise: questions only one of two styles solves (p>0)
    exclusive = {}
    for i, a in enumerate(STYLES):
        for b in STYLES[i + 1 :]:
            sa, sb = wide[a] > 0, wide[b] > 0
            exclusive[f"{a}_only_vs_{b}"] = int((sa & ~sb).sum())
            exclusive[f"{b}_only_vs_{a}"] = int((sb & ~sa).sum())
            exclusive[f"{a}_and_{b}"] = int((sa & sb).sum())

    union_solved = int((wide > 0).any(axis=1).sum())
    return {
        "benchmark": bench,
        "n_questions": n_q,
        "n_samples": 256,
        "n_unsolved_by_all_styles": n_unsolved,
        "n_unique_winner": n_unique,
        "n_tied_winner": n_tied,
        "union_solved_pgt0": union_solved,
        "mean_p_success": mean_p,
        "n_questions_with_pgt0": n_solved_by,
        "unique_best_count": unique_wins,
        "unique_best_pct": {s: unique_wins[s] / n_q * 100.0 for s in STYLES},
        "best_including_ties_count": incl_ties,
        "best_including_ties_pct": {s: incl_ties[s] / n_q * 100.0 for s in STYLES},
        "unique_best_by_subject": {k: dict(v) for k, v in unique_by_subject.items()},
        "largest_gaps": gaps[:25],
        "pairwise_pgt0": exclusive,
        "per_question": [
            {
                "question_id": qid,
                "subject": subject.loc[qid],
                "p_success": {s: float(wide.loc[qid, s]) for s in STYLES},
                "n_correct": {s: int(n_corr.loc[qid, s]) for s in STYLES},
                "best": [STYLES[j] for j in range(k) if is_max[i, j]] if any_correct[i] else [],
                "unique_best": (
                    STYLES[int(np.argmax(P[i]))]
                    if any_correct[i] and n_at_max[i] == 1
                    else None
                ),
            }
            for i, qid in enumerate(qids)
        ],
    }


def markdown(summary: dict) -> str:
    bench = summary["benchmark"]
    n = summary["n_questions"]
    lines = [
        f"# {bench} · 0.6B-Base style_i SFT · which style is best per question",
        "",
        "Each column is a **different checkpoint** (prefix-free SFT on that GMM style only),",
        "valbest, 256 samples, T=1, max 4096. `p` = fraction of the 256 completions that grade correct.",
        "A style is the **unique best** if it has strictly highest `p` and `p>0`.",
        "All-zero questions (no style ever correct) are not wins. Ties are counted separately.",
        "",
        f"Questions: **{n}**. Unsolved by every style: **{summary['n_unsolved_by_all_styles']}**. "
        f"Unique winner: **{summary['n_unique_winner']}**. Tied winners: **{summary['n_tied_winner']}**. "
        f"Union with at least one correct sample: **{summary['union_solved_pgt0']}**.",
        "",
        "| Style | mean p (Pass@1) | questions with p>0 | unique best | unique % | best incl. ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in STYLES:
        lines.append(
            f"| {s} | {100*summary['mean_p_success'][s]:.2f}% | "
            f"{summary['n_questions_with_pgt0'][s]} | "
            f"{summary['unique_best_count'][s]} | "
            f"{summary['unique_best_pct'][s]:.1f}% | "
            f"{summary['best_including_ties_count'][s]} |"
        )
    if summary["unique_best_by_subject"] and bench.startswith("MATH"):
        lines += ["", "## Unique best by MATH subject", "", "| Subject | " + " | ".join(STYLES) + " |", "|---|" + "---:|" * 6]
        for subj, counts in sorted(summary["unique_best_by_subject"].items()):
            lines.append("| " + subj + " | " + " | ".join(str(counts[s]) for s in STYLES) + " |")
    lines += [
        "",
        "## Largest unique-best gaps",
        "",
        "Questions where one style is strictly ahead of the runner-up (p_best − p_second).",
        "",
        "| question | subject | best | p_best | second | p_second | gap | s1..s6 n_correct/256 |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for g in summary["largest_gaps"][:15]:
        nc = g["n_correct"]
        nc_s = " ".join(str(nc[s]) for s in STYLES)
        q = str(g["question_id"]).replace("|", "/")
        lines.append(
            f"| `{q}` | {g['subject']} | {g['best']} | {100*g['p_best']:.1f}% | "
            f"{g['second']} | {100*g['p_second']:.1f}% | {100*g['gap']:.1f}pp | {nc_s} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_claims" / "style_i_question_wins_0p6b",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summ = {}
    for bench in ("MATH-500", "AMC12-2025"):
        df = load_bench(bench)
        summ = summarize(df)
        all_summ[bench] = {k: v for k, v in summ.items() if k != "per_question"}
        stem = "math500" if bench == "MATH-500" else "amc12"
        (args.out_dir / f"{stem}_per_question.csv").write_text(
            pd.DataFrame(
                [
                    {
                        "question_id": r["question_id"],
                        "subject": r["subject"],
                        "unique_best": r["unique_best"],
                        "best": ",".join(r["best"]),
                        **{f"p_{s}": r["p_success"][s] for s in STYLES},
                        **{f"n_{s}": r["n_correct"][s] for s in STYLES},
                    }
                    for r in summ["per_question"]
                ]
            ).to_csv(index=False)
        )
        (args.out_dir / f"{stem}_summary.json").write_text(
            json.dumps({k: v for k, v in summ.items() if k != "per_question"}, indent=2)
        )
        md = markdown(summ)
        (args.out_dir / f"{stem}.md").write_text(md)
        print(md)
        print(f"wrote {args.out_dir / f'{stem}.md'}", flush=True)

    (args.out_dir / "summary.json").write_text(json.dumps(all_summ, indent=2))
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
