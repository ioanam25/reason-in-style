#!/usr/bin/env python3
"""Build PHASE_AB_EOSFIX_STYLE_TABLE.md from style_claims_registry.json.

Run locally after syncing the registry from scratch:
  scp <cluster>:.../style_claims/style_claims_registry.json .
  python scripts/analysis/render_style_claims_table.py --registry style_claims_registry.json

Or on-cluster:
  python scripts/analysis/render_style_claims_table.py
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

DEFAULT_REGISTRY = Path(
    os.environ.get("ARCHIVE_ROOT", "scratch/archive") + "/style_claims/style_claims_registry.json"
)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "PHASE_AB_EOSFIX_STYLE_TABLE.md"

BENCH_ORDER = {"math500": 0, "amc": 1, "olympiad": 2}
STUDENT_ORDER = {
    "4B-Thinking": 0,
    "4B-Base": 1,
    "4B-Instruct": 2,
    "1.7B": 3,
    "1.7B-Base": 4,
    "0.6B": 5,
    "0.6B-Base": 6,
}
ARM_ORDER = {
    "vanilla SFT": 0,
    "AE-GMM K=6": 1,
    "A1 random K=6": 2,
    "A2 constant K=1": 3,
    "A3 ModC K=9": 4,
    "B1 class-balanced GMM": 5,
    "B2 special tokens": 6,
    "B3 descriptors": 7,
}


def sort_key(row: dict):
    return (
        BENCH_ORDER.get(row.get("benchmark", ""), 9),
        STUDENT_ORDER.get(row.get("student", ""), 9),
        ARM_ORDER.get(row.get("arm", ""), 9),
    )


def pct(x, nd=1):
    if x is None:
        return "—"
    return f"{100 * float(x):.{nd}f}%"


def f3(x):
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def f2(x):
    if x is None:
        return "—"
    return f"{float(x):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--teacher-profiles", type=Path, default=None)
    args = ap.parse_args()

    rows = json.loads(args.registry.read_text())
    rows = sorted(rows, key=sort_key)
    eos = [r for r in rows if r.get("protocol") == "eosfix"]

    lines = [
        "# Phase A/B style claims (EOS_FIX protocol)",
        "",
        f"Generated from `{args.registry}` ({len(eos)} eval dirs).",
        "",
        "All forced-style geometry, identifiability, intended-vs-realized, and complementarity",
        "numbers below come from **untruncated** EOS_FIX generations (`finish_reason` recorded).",
        "",
    ]

    # --- forced style (tab:forced rebuild) ---
    lines += [
        "## Forced-style controllability (`tab:forced` rebuild)",
        "",
        "Matched-question metrics across all prefix styles. Len gap = mean pairwise relative",
        "token-length gap; TF-IDF = matched mean pairwise cosine distance; Collapse = mean of",
        "length-CV and TF-IDF collapse rates (<5% separation).",
        "",
        "| Arm | Student | Bench | Pass@256 | TF-IDF | Collapse | Len gap | trunc% |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    legacy_math500 = {
        ("4B-Base", "AE-GMM K=6"): {"tfidf": 0.21, "collapse": 0.48, "len": 0.041},
        ("4B-Instruct", "AE-GMM K=6"): {"tfidf": 0.17, "collapse": 0.47, "len": 0.047},
        ("4B-Thinking", "AE-GMM K=6"): {"tfidf": 0.09, "collapse": 0.63, "len": 0.016},
        ("1.7B-Base", "AE-GMM K=6"): {"tfidf": 0.51, "collapse": 0.34, "len": 0.058},
    }
    for r in eos:
        if r.get("n_styles", 0) < 2:
            continue
        forced = r.get("forced", {})
        p256 = (r.get("pass_at_k") or {}).get("256")
        trunc = r.get("pct_finish_length")
        lines.append(
            f"| {r['arm']} | {r['student']} | {r['benchmark']} | "
            f"{pct(p256 / 100 if p256 else None)} | "
            f"{f3(forced.get('tfidf_dist'))} | "
            f"{f2(forced.get('collapse_score'))} | "
            f"{f2((forced.get('len_sep_pct') or 0) / 100 if forced.get('len_sep_pct') is not None else None)} | "
            f"{f2(trunc)} |"
        )

    lines += ["", "### Legacy vs EOS_FIX (MATH-500 AE-GMM only)", ""]
    lines += [
        "| Student | Legacy TF-IDF | EOS TF-IDF | Δ | Legacy len | EOS len | Legacy trunc | EOS trunc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for student, leg in legacy_math500.items():
        st, arm = student
        new = next(
            (
                x
                for x in eos
                if x.get("student") == st and x.get("arm") == arm and x.get("benchmark") == "math500"
            ),
            None,
        )
        if not new:
            continue
        f = new.get("forced", {})
        new_tfidf = f.get("tfidf_dist")
        new_len = (f.get("len_sep_pct") or 0) / 100 if f.get("len_sep_pct") is not None else None
        delta = None if new_tfidf is None else new_tfidf - leg["tfidf"]
        lines.append(
            f"| {st} | {leg['tfidf']:.2f} | {f3(new_tfidf)} | {f3(delta)} | "
            f"{leg['len']:.3f} | {f2(new_len)} | ~93%* | {f2(new.get('pct_finish_length'))} |"
        )
    lines += ["", "*Legacy 4B-Base trunc rate from draft header; others similar on truncated runs.", ""]

    # --- identifiability ---
    lines += [
        "## Style identifiability (prefix recoverable from generation?)",
        "",
        "Question-level split. Chance = 1/K. Best = max(handcrafted RF, TF-IDF logreg).",
        "",
        "| Arm | Student | Bench | K | Chance | Best acc | Δ vs chance | HC-RF | TF-IDF | trunc% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eos:
        ident = r.get("identifiability") or {}
        k = ident.get("n_styles") or r.get("n_styles")
        if not k or k < 2:
            continue
        chance = ident.get("chance") or (1.0 / k)
        best = ident.get("best_acc")
        hc = (ident.get("handcrafted_full") or {}).get("acc")
        tf = (ident.get("tfidf_logreg") or {}).get("acc")
        lines.append(
            f"| {r['arm']} | {r['student']} | {r['benchmark']} | {k} | {pct(chance)} | "
            f"{pct(best)} | {pct((best - chance) if best is not None else None)} | "
            f"{pct(hc)} | {pct(tf)} | {f2(r.get('pct_finish_length'))} |"
        )

    # --- intended vs realized ---
    lines += [
        "",
        "## Intended vs realized style (diagonal = prefix matches nearest teacher centroid)",
        "",
        "| Arm | Student | Bench | Diagonal | Chance | Δ | Realized H (bits) | Profile rank ρ |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in eos:
        real = r.get("realized") or {}
        if not real:
            continue
        diag = real.get("diagonal_mass")
        chance = real.get("chance_diagonal")
        lines.append(
            f"| {r['arm']} | {r['student']} | {r['benchmark']} | {pct(diag)} | {pct(chance)} | "
            f"{pct(real.get('diagonal_over_chance'))} | "
            f"{f2(real.get('realized_entropy_bits'))} | {f3(real.get('profile_rank_corr_mean'))} |"
        )

    # --- complementarity ---
    lines += [
        "",
        "## Style complementarity (why Pass@k rises)",
        "",
        "| Arm | Student | Bench | Pass@128 all | Max LOSO Δ | Mean Jaccard | Union cov | Unique solves (best style) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in eos:
        comp = r.get("complementarity") or {}
        if comp.get("note"):
            continue
        uniq = comp.get("unique_solves") or {}
        best_uniq = max(uniq.values()) if uniq else None
        lines.append(
            f"| {r['arm']} | {r['student']} | {r['benchmark']} | "
            f"{pct((r.get('pass_at_k') or {}).get('128') / 100 if (r.get('pass_at_k') or {}).get('128') else None)} | "
            f"{pct(comp.get('max_loso_delta'))} | "
            f"{f3(comp.get('mean_pairwise_jaccard'))} | "
            f"{pct(comp.get('union_coverage'))} | {best_uniq or '—'} |"
        )

    # --- arm controllability summary ---
    lines += [
        "",
        "## Arm controllability ablation (MATH-500 headline students)",
        "",
        "Identifiability best-acc aggregated across benchmarks where available.",
        "",
        "| Arm | 4B-Base ident | 4B-Instruct ident | 4B-Thinking ident | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    ctrl_arms = [
        "AE-GMM K=6",
        "A1 random K=6",
        "A2 constant K=1",
        "A3 ModC K=9",
        "B1 class-balanced GMM",
        "B2 special tokens",
        "B3 descriptors",
        "vanilla SFT",
    ]
    for arm in ctrl_arms:
        cells = []
        for st in ("4B-Base", "4B-Instruct", "4B-Thinking"):
            vals = [
                (x.get("identifiability") or {}).get("best_acc")
                for x in eos
                if x.get("arm") == arm and x.get("student") == st
            ]
            vals = [v for v in vals if v is not None]
            cells.append(pct(sum(vals) / len(vals)) if vals else "—")
        note = ""
        if arm == "A1 random K=6":
            note = "critical null: random prefix content"
        elif arm == "B3 descriptors":
            note = "natural-language style descriptors"
        elif arm == "vanilla SFT":
            note = "pseudo-style null"
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {note} |")

    # teacher profiles if present
    tp = args.teacher_profiles
    if tp is None:
        tp = args.registry.parent / "teacher_style_profiles.json"
    if tp.exists():
        prof = json.loads(tp.read_text())
        tag = "covz-qwen3-4b-final"
        if tag in prof:
            p = prof[tag]["profile"]
            lines += [
                "",
                f"## Teacher-side profile ({tag}, GMM K=6)",
                "",
                "| Feature | " + " | ".join(f"s{i+1}" for i in range(6)) + " |",
                "|---|" + "|".join(["---:"] * 6) + "|",
            ]
            for feat in ("n_words", "dens_backtrack", "dens_verification", "dens_equals", "lines_per_100w"):
                vals = [f"{p[f'style_{i+1}'][feat]:.2f}" for i in range(6)]
                lines.append(f"| {feat} | " + " | ".join(vals) + " |")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
