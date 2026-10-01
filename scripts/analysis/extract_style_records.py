#!/usr/bin/env python3
"""Stage 1: stream each EOS_FIX generations dump once into compact style records.

The dumps total ~90 GB, so every downstream style analysis reads these artifacts
instead of the raw jsonl:

  <out-root>/<eval_dir>/records.parquet   one row per sample: ids, correctness,
                                          finish_reason, and the handcrafted
                                          style-feature vector
  <out-root>/<eval_dir>/texts.jsonl       reservoir sample of truncated outputs,
                                          per prefix style (for TF-IDF probes)
  <out-root>/<eval_dir>/meta.json         arm / student / benchmark + row counts

Usage (array job shards over the tracked eval dirs):
  python scripts/analysis/extract_style_records.py --shard 0 --num-shards 70 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.handcrafted_style_features import FEATURE_NAMES, extract_features  # noqa: E402
from scripts.style_registry import EVAL_ROOT, iter_eval_dirs  # noqa: E402

DEFAULT_OUT = Path(os.environ.get("ARCHIVE_ROOT", "scratch/archive")) / "style_records"
TEXT_CLIP = 6000
# Keep every Nth question (by stable hash) rather than every Nth row, so the text
# sample holds all styles for the questions it covers. The matched-question
# TF-IDF metric and the question-level identifiability split both need that.
TEXT_QUESTION_MOD = 4
TEXT_PER_QUESTION_STYLE = 4
FEATURE_BLOCK = 50_000


def _keep_question(qid: str, mod: int) -> bool:
    return zlib.crc32(qid.encode("utf-8")) % mod == 0


def _parse_line(payload: tuple[int, str, int]):
    """Worker: parse one dump row and score it in handcrafted feature space."""
    _idx, line, qmod = payload
    try:
        o = json.loads(line)
    except Exception:
        return None
    text = o.get("output") or o.get("text") or ""
    feats = extract_features(text)
    qid = str(o.get("question_id"))
    keep_text = text[:TEXT_CLIP] if _keep_question(qid, qmod) else None
    return (
        qid,
        o.get("prefix_style"),
        bool(o.get("correct")),
        int(o.get("n_tokens") or 0),
        o.get("finish_reason"),
        int(o.get("output_len") or len(text)),
        [feats[k] for k in FEATURE_NAMES],
        keep_text,
    )


class QuestionMatchedSampler:
    """Up to `per_cell` texts for each (question, style) on the sampled questions."""

    def __init__(self, per_cell: int):
        self.per_cell = per_cell
        self.cells: dict[tuple[str, str], list[str]] = {}

    def add(self, qid: str, style: str, text: str) -> None:
        cell = self.cells.setdefault((qid, style), [])
        if len(cell) < self.per_cell:
            cell.append(text)

    def rows(self):
        for (qid, style), texts in sorted(self.cells.items()):
            for t in texts:
                yield {"question_id": qid, "prefix_style": style, "text": t}


def process_dir(d: Path, meta: dict, out_root: Path, workers: int, qmod: int, per_cell: int) -> dict:
    out_dir = out_root / d.name
    out_dir.mkdir(parents=True, exist_ok=True)
    gens = d / "generations.jsonl"

    qids: list[str] = []
    styles: list[str] = []
    corrects: list[bool] = []
    ntoks: list[int] = []
    finishes: list[str] = []
    olens: list[int] = []
    # Feature rows are flushed to float32 blocks so a 2.5 GB dump does not need a
    # 170k-element list of Python floats resident at once.
    feat_buf: list[list[float]] = []
    feat_blocks: list[np.ndarray] = []
    sampler = QuestionMatchedSampler(per_cell)

    def flush_features() -> None:
        if feat_buf:
            feat_blocks.append(np.asarray(feat_buf, dtype=np.float32))
            feat_buf.clear()

    with gens.open() as fh, Pool(workers, maxtasksperchild=2000) as pool:
        stream = ((i, line, qmod) for i, line in enumerate(fh))
        for rec in pool.imap(_parse_line, stream, chunksize=256):
            if rec is None:
                continue
            qid, style, correct, ntok, finish, olen, feats, keep_text = rec
            qids.append(sys.intern(qid))
            styles.append(sys.intern(style if style is not None else ""))
            corrects.append(correct)
            ntoks.append(ntok)
            finishes.append(sys.intern(finish if finish is not None else ""))
            olens.append(olen)
            feat_buf.append(feats)
            if len(feat_buf) >= FEATURE_BLOCK:
                flush_features()
            if keep_text is not None:
                sampler.add(qid, style if style is not None else "", keep_text)
    flush_features()

    if not qids:
        raise RuntimeError(f"no rows parsed from {gens}")

    df = pd.DataFrame(
        {
            "question_id": pd.Categorical(qids),
            "prefix_style": pd.Categorical(styles),
            "correct": np.asarray(corrects, dtype=bool),
            "n_tokens": np.asarray(ntoks, dtype=np.int32),
            "finish_reason": pd.Categorical(finishes),
            "output_len": np.asarray(olens, dtype=np.int32),
        }
    )
    feats = np.concatenate(feat_blocks, axis=0)
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = feats[:, i]
    df.to_parquet(out_dir / "records.parquet", index=False)

    n_text_rows = 0
    with (out_dir / "texts.jsonl").open("w") as fh:
        for row in sampler.rows():
            fh.write(json.dumps(row) + "\n")
            n_text_rows += 1

    n_styles = sorted({s for s in styles if s})
    info = dict(meta)
    info.update(
        {
            "n_rows": len(qids),
            "n_questions": int(df["question_id"].nunique()),
            "styles": n_styles,
            "n_styles": len(n_styles),
            "pct_finish_length": round(100.0 * float(np.mean([f == "length" for f in finishes])), 2),
            "text_question_mod": qmod,
            "text_per_question_style": per_cell,
            "n_text_rows": n_text_rows,
            "n_text_questions": len({q for q, _ in sampler.cells}),
            "feature_names": FEATURE_NAMES,
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--protocol", default="eosfix")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--text-question-mod", type=int, default=TEXT_QUESTION_MOD)
    ap.add_argument("--text-per-question-style", type=int, default=TEXT_PER_QUESTION_STYLE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="only extract dirs whose name contains this (repeatable, AND)",
    )
    ap.add_argument("--list", action="store_true", help="print the tracked dirs and exit")
    args = ap.parse_args()

    targets = list(iter_eval_dirs(args.eval_root, args.protocol))
    if args.name_contains:
        targets = [
            (d, m) for d, m in targets if all(s in d.name for s in args.name_contains)
        ]
    targets.sort(key=lambda t: t[0].name)

    if args.list:
        for d, meta in targets:
            print(f"{meta['arm']:24} {meta['student']:12} {meta['benchmark']:9} {d.name}")
        print(f"total={len(targets)}")
        return

    mine = targets[args.shard :: args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} dirs, workers={args.workers}", flush=True)
    for d, meta in mine:
        out_dir = args.out_root / d.name
        if (out_dir / "records.parquet").exists() and not args.force:
            print(f"skip (exists) {d.name}", flush=True)
            continue
        print(f"extract {d.name} [{meta['arm']} / {meta['student']} / {meta['benchmark']}]", flush=True)
        info = process_dir(
            d,
            meta,
            args.out_root,
            args.workers,
            args.text_question_mod,
            args.text_per_question_style,
        )
        print(
            f"  rows={info['n_rows']} q={info['n_questions']} styles={info['n_styles']} "
            f"trunc={info['pct_finish_length']}% textq={info['n_text_questions']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
