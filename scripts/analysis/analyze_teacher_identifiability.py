#!/usr/bin/env python3
"""Teacher-side identifiability: can GMM style (or oracle teacher ID) be read
from the CoT text, with a question-held-out split?

This is the ceiling for student prefix recovery. Near-chance GMM recovery means
the partition is not a textual style even before SFT. High GMM recovery + chance
student recovery means transfer failed. Oracle teacher-ID recovery is the
Lippmann-style ceiling (K=9, chance 1/9).

Usage (on the training cluster):
  python scripts/analysis/analyze_teacher_identifiability.py
"""

from __future__ import annotations
import os

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.analyze_style_claims import identifiability_block  # noqa: E402
from scripts.handcrafted_style_features import FEATURE_NAMES, extract_features  # noqa: E402

SROOT = Path(os.environ.get("ARCHIVE_ROOT", os.environ.get("SROOT", "scratch/archive")))
DEFAULT_PARQUET = SROOT / "cluster_prefix" / "covz-qwen3-4b-final-gmm" / "k6" / "train.parquet"
DEFAULT_OUT = SROOT / "style_claims" / "teacher_identifiability"
TEXT_CLIP = 6000


def assistant_text(messages) -> str:
    if messages is None:
        return ""
    rows = messages.tolist() if hasattr(messages, "tolist") else messages
    for m in rows:
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def _one(payload: tuple[str, str, str]) -> tuple[dict, dict]:
    qid, style, text = payload
    feats = extract_features(text)
    feats["n_tokens"] = feats.get("n_words", 0.0)
    rec = {
        "question_id": qid,
        "prefix_style": style,
        **{k: feats[k] for k in FEATURE_NAMES},
        "n_tokens": feats["n_tokens"],
    }
    tex = {"question_id": qid, "prefix_style": style, "text": text[:TEXT_CLIP]}
    return rec, tex


def cap_questions(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    qids = df["question_id"].astype(str).unique()
    rng.shuffle(qids)
    n_by_q = df.groupby(df["question_id"].astype(str)).size()
    keep, n = [], 0
    for q in qids:
        keep.append(q)
        n += int(n_by_q.get(q, 0))
        if n >= max_rows:
            break
    return df[df["question_id"].astype(str).isin(keep)].copy()


def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = [
        (str(qid), str(style), assistant_text(raw))
        for qid, style, raw in zip(df["question_id"], df["label"], df["messages"])
    ]
    with ProcessPoolExecutor() as pool:
        got = list(pool.map(_one, payloads, chunksize=256))
    rec = pd.DataFrame([r for r, _ in got])
    tex = pd.DataFrame([t for _, t in got])
    return rec, tex


def run_target(df: pd.DataFrame, col: str, seed: int, max_rows: int) -> dict:
    sub = df.rename(columns={col: "label"})[["question_id", "label", "messages"]].copy()
    sub = sub[sub["label"].astype(str).str.len() > 0]
    sub = cap_questions(sub, max_rows, seed)
    print(f"  {col}: featurizing {len(sub)} traces / {sub['question_id'].nunique()} questions", flush=True)
    styles = sorted(sub["label"].astype(str).unique().tolist())
    rec, tex = featurize(sub)
    ident = identifiability_block(rec, tex, styles, seed, max_rows)
    rng = np.random.default_rng(seed + 1)
    rec_sh = rec.copy()
    rec_sh["prefix_style"] = rng.permutation(rec_sh["prefix_style"].to_numpy())
    tex_sh = tex.copy()
    tex_sh["prefix_style"] = rng.permutation(tex_sh["prefix_style"].to_numpy())
    ident["shuffled_label"] = identifiability_block(rec_sh, tex_sh, styles, seed, max_rows)
    ident["n_traces"] = int(len(sub))
    ident["n_questions"] = int(sub["question_id"].nunique())
    ident["label_counts"] = sub["label"].astype(str).value_counts().to_dict()
    return ident


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-probe-rows", type=int, default=60000)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["messages", "question_id", "style_name", "oracle_style_name"])
    print(f"loaded {len(df)} traces from {args.parquet}", flush=True)

    out = {
        "source": str(args.parquet),
        "n_traces": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "seed": args.seed,
        "max_probe_rows": args.max_probe_rows,
        "split": "question-held-out 70/30, same probe as student identifiability",
        "gmm_style": run_target(df, "style_name", args.seed, args.max_probe_rows),
        "oracle_teacher": run_target(df, "oracle_style_name", args.seed, args.max_probe_rows),
        "updated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    js = args.out_dir / "teacher_identifiability.json"
    js.write_text(json.dumps(out, indent=2) + "\n")

    def line(title: str, block: dict) -> list[str]:
        k = block.get("n_styles")
        chance = block.get("chance")
        best = block.get("best_acc")
        rows = [
            f"## {title}",
            f"- traces {block.get('n_traces')} / questions {block.get('n_questions')} / K={k} chance={chance:.3f}"
            if chance is not None
            else f"- traces {block.get('n_traces')}",
            f"- best acc {None if best is None else f'{100 * best:.1f}%'} "
            f"(+{100 * (block.get('best_acc_over_chance') or 0):.1f} pp vs chance)",
        ]
        for nm in ("tfidf_logreg", "handcrafted_full", "handcrafted_density", "length_only"):
            if nm in block:
                rows.append(f"- {nm}: {100 * block[nm]['acc']:.1f}% acc, macro-F1 {block[nm]['macro_f1']:.3f}")
        sh = block.get("shuffled_label") or {}
        if sh.get("best_acc") is not None:
            rows.append(f"- shuffled-label best acc {100 * sh['best_acc']:.1f}%")
        rows.append("")
        return rows

    md = [
        "# Teacher-side identifiability",
        "",
        f"Source: `{args.parquet}`",
        "Question-held-out probes, same classifiers as student prefix recovery.",
        "",
    ]
    md += line("GMM style_i (K=6)", out["gmm_style"])
    md += line("Oracle teacher ID (ModC, K=9)", out["oracle_teacher"])
    md.append("If GMM ≈ chance, the partition is not readable in the CoT. If GMM ≫ chance but students stay at 1/6, transfer failed.")
    md.append("")
    (args.out_dir / "teacher_identifiability.md").write_text("\n".join(md))
    print(f"wrote {js}", flush=True)
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
