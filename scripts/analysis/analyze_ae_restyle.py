#!/usr/bin/env python3
"""Score AE restyle generations: intended μ_j vs realized handcrafted style + answer match.

Reads qualitative_decodes.json from swap_style_latent.py (greedy [c(s); μ_j]
decodes). For each decode this reports:

  profile     mean handcrafted features per intended style, vs teacher GMM centroids
  realized    nearest teacher centroid of the decode; KxK intended-vs-realized
  answers     boxed-answer match vs the original teacher trace (identity and each μ_j)

This is the generation-side test of whether z is a usable style handle. NLL swap
already lives in swap_matrix.json; this script does not recompute NLL.

Usage:
  python scripts/analysis/analyze_ae_restyle.py \\
    --decodes <SCRATCH_ROOT>/.../latent_swap/covz-qwen3-4b-final-restyle/qualitative_decodes.json
"""

from __future__ import annotations
import os

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_style_claims import (  # noqa: E402
    assign_nearest_style,
    load_teacher_basis,
)
from scripts.handcrafted_style_features import PROFILE_FEATURES, extract_features  # noqa: E402
from scripts.math_answer_utils import answers_match, parse_boxed_answer  # noqa: E402

DEFAULT_DECODES = Path(
    os.environ.get("ARCHIVE_ROOT", "scratch/archive") + "/latent_swap/"
    "covz-qwen3-4b-final/qualitative_decodes.json"
)
DEFAULT_GMM_PREFIX = Path(
    os.environ.get("ARCHIVE_ROOT", "scratch/archive") + "/cluster_prefix/"
    "covz-qwen3-4b-final-gmm/k6"
)
TEACHER_FEATURES_NPZ = REPO_ROOT / "data/scas/cluster_handcrafted_sweep-qwen3-0p6b/handcrafted_features.npz"
PROFILE_FOCUS = [
    "n_words",
    "lines_per_100w",
    "dens_verification",
    "dens_backtrack",
    "dens_equals",
]


def _qid_keys(qid: str, source: str | None = None) -> list[str]:
    keys = [qid]
    if source:
        keys.append(f"{source}::{qid}")
    if "::" in qid:
        keys.append(qid.split("::", 1)[1])
    return keys


def _load_gold_from_prefix(prefix_dir: Path) -> dict[tuple[str, str], str]:
    gold: dict[tuple[str, str], str] = {}
    for split in ("train", "validation"):
        f = prefix_dir / f"{split}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["question_id", "oracle_style_name", "source_dataset", "answer"])
        for q, src, t, a in zip(
            df["question_id"].astype(str),
            df["source_dataset"].astype(str),
            df["oracle_style_name"].astype(str),
            df["answer"].astype(str),
        ):
            for key in _qid_keys(q, src):
                gold[(key, t)] = a
    return gold


def _match(pred_text: str, gold: str) -> bool:
    if not gold:
        return False
    boxed = parse_boxed_answer(pred_text)
    if boxed and answers_match(pred_text, gold):
        return True
    if boxed and answers_match(boxed, gold):
        return True
    return bool(pred_text) and answers_match(pred_text, gold)


def _features_frame(texts: list[str]) -> pd.DataFrame:
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(decodes: dict, basis: dict, gold_lookup: dict[tuple[str, str], str]) -> dict:
    styles = list(basis["styles"])
    samples = decodes.get("samples") or []
    if not samples:
        raise SystemExit("no samples in qualitative_decodes.json")

    intended, realized, n_words, has_boxed = [], [], [], []
    gold_hits = {s: [] for s in styles}
    gold_hits["identity"] = []
    ident_n_words = []

    per_style_feats: dict[str, list[dict]] = {s: [] for s in styles}
    confusion = np.zeros((len(styles), len(styles)), dtype=np.int64)

    n_gold = 0
    for smp in samples:
        qid = str(smp.get("question_id"))
        teacher = str(smp.get("teacher_style_name") or "")
        gold = smp.get("gold_answer") or ""
        if not gold:
            for key in _qid_keys(qid):
                gold = gold_lookup.get((key, teacher), "")
                if gold:
                    break
        if not gold and smp.get("teacher_trace"):
            gold = parse_boxed_answer(smp["teacher_trace"])
        if gold:
            n_gold += 1

        ident = smp.get("identity_decode") or ""
        ident_n_words.append(extract_features(ident)["n_words"])
        gold_hits["identity"].append(_match(ident, gold) if gold else None)

        decs = smp.get("decodes") or {}
        texts = [decs.get(s, "") for s in styles]
        feats = _features_frame(texts)
        assigned = assign_nearest_style(feats, basis)
        for j, s in enumerate(styles):
            intended.append(s)
            realized.append(str(assigned[j]))
            n_words.append(float(feats.iloc[j]["n_words"]))
            boxed = "\\boxed" in (texts[j] or "")
            has_boxed.append(boxed)
            per_style_feats[s].append(feats.iloc[j].to_dict())
            gold_hits[s].append(_match(texts[j], gold) if gold else None)
            confusion[j, styles.index(str(assigned[j]))] += 1

    decode_profile = {
        s: {f: float(np.mean([row[f] for row in per_style_feats[s]])) for f in PROFILE_FEATURES}
        for s in styles
    }
    teacher_profile = basis["profile"]

    rank_corr = {}
    for f in PROFILE_FOCUS:
        dec = np.array([decode_profile[s][f] for s in styles], dtype=np.float64)
        tea = np.array([teacher_profile[s][f] for s in styles], dtype=np.float64)
        rank_corr[f] = spearman(dec, tea)

    n_int = len(intended)
    acc = float(np.mean([a == b for a, b in zip(intended, realized)])) if n_int else 0.0
    chance = 1.0 / max(1, len(styles))

    def _rate(vals):
        xs = [v for v in vals if v is not None]
        return float(np.mean(xs)) if xs else None

    answer_rates = {s: _rate(gold_hits[s]) for s in ["identity"] + styles}

    # length should rise style_1 -> style_6 if z carries the teacher length axis
    mean_nw = [decode_profile[s]["n_words"] for s in styles]
    length_mono = float(np.mean([mean_nw[i] <= mean_nw[i + 1] for i in range(len(mean_nw) - 1)]))

    summary = {
        "n_questions": len(samples),
        "n_with_gold": int(n_gold),
        "max_new_tokens": decodes.get("max_new_tokens"),
        "tag": decodes.get("tag") or basis["tag"],
        "k": len(styles),
        "styles": styles,
        "decode_profile": decode_profile,
        "teacher_profile": {s: {f: teacher_profile[s][f] for f in PROFILE_FOCUS} for s in styles},
        "profile_rank_corr_vs_teacher": rank_corr,
        "mean_n_words_by_intended": {s: decode_profile[s]["n_words"] for s in styles},
        "identity_mean_n_words": float(np.mean(ident_n_words)),
        "length_monotonicity_adjacent": length_mono,
        "pct_has_boxed_by_intended": {
            s: float(np.mean([has_boxed[i] for i, t in enumerate(intended) if t == s]))
            for s in styles
        },
        "realized": {
            "accuracy": acc,
            "chance": chance,
            "delta_vs_chance": acc - chance,
            "confusion_intended_by_realized": {
                styles[i]: {styles[j]: int(confusion[i, j]) for j in range(len(styles))}
                for i in range(len(styles))
            },
            "realized_mass": dict(Counter(realized)),
        },
        "answer_match": answer_rates,
        "answer_match_drop_vs_identity": {
            s: (
                None
                if answer_rates["identity"] is None or answer_rates[s] is None
                else float(answer_rates[s] - answer_rates["identity"])
            )
            for s in styles
        },
    }
    return summary


def render_md(summary: dict) -> str:
    styles = summary["styles"]
    lines = [
        "# AE restyle (generation)",
        "",
        f"n_questions = {summary['n_questions']}  ·  gold = {summary['n_with_gold']}  "
        f"·  max_new_tokens = {summary.get('max_new_tokens')}",
        "",
        "## Intended vs realized (nearest teacher centroid)",
        "",
        f"accuracy = {100 * summary['realized']['accuracy']:.1f}%  "
        f"(chance {100 * summary['realized']['chance']:.1f}%)",
        "",
        "## Profile means and rank correlation vs teacher",
        "",
        "| intended | n_words | lines/100w | dens_verif | dens_back | dens_= |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    dp = summary["decode_profile"]
    for s in styles:
        lines.append(
            f"| {s} | {dp[s]['n_words']:.0f} | {dp[s]['lines_per_100w']:.1f} | "
            f"{dp[s]['dens_verification']:.2f} | {dp[s]['dens_backtrack']:.2f} | "
            f"{dp[s]['dens_equals']:.1f} |"
        )
    lines.append("")
    lines.append("Spearman(decode rank, teacher rank):")
    for f, v in summary["profile_rank_corr_vs_teacher"].items():
        lines.append(f"- `{f}`: {v:.3f}" if v == v else f"- `{f}`: nan")
    lines += [
        "",
        f"adjacent length monotonicity (style_1 ≤ … ≤ style_6): "
        f"{100 * summary['length_monotonicity_adjacent']:.0f}% of adjacent pairs",
        "",
        "## Answer match vs teacher gold",
        "",
        "| decode | match | Δ vs identity |",
        "|---|---:|---:|",
    ]
    ident = summary["answer_match"].get("identity")
    ident_s = "—" if ident is None else f"{100 * ident:.1f}%"
    lines.append(f"| identity [c;z] | {ident_s} | — |")
    for s in styles:
        m = summary["answer_match"][s]
        d = summary["answer_match_drop_vs_identity"][s]
        ms = "—" if m is None else f"{100 * m:.1f}%"
        ds = "—" if d is None else f"{100 * d:+.1f} pp"
        lines.append(f"| μ {s} | {ms} | {ds} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decodes", type=Path, default=DEFAULT_DECODES)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--features-npz", type=Path, default=TEACHER_FEATURES_NPZ)
    ap.add_argument("--gmm-prefix", type=Path, default=DEFAULT_GMM_PREFIX)
    ap.add_argument("--tag", default="covz-qwen3-4b-final")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    decodes = json.loads(args.decodes.read_text())
    tag = decodes.get("tag") or args.tag
    out_dir = args.out_dir or args.decodes.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    basis = load_teacher_basis(tag, args.k, args.features_npz)
    gold = _load_gold_from_prefix(args.gmm_prefix) if args.gmm_prefix.exists() else {}
    print(f"gold lookup rows: {len(gold)}", flush=True)

    summary = analyze(decodes, basis, gold)
    (out_dir / "restyle_summary.json").write_text(json.dumps(summary, indent=2))
    md = render_md(summary)
    (out_dir / "RESTYLE_SUMMARY.md").write_text(md)
    print(md)
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
