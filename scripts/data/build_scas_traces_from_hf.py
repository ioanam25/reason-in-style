#!/usr/bin/env python3
"""Build on-disk ``scas_traces_dataset`` for AE / z-extract / GMM from Hugging Face.

Source:
  https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool

Writes a HuggingFace ``DatasetDict`` with ``train`` / ``validation`` splits
(split by question, no leakage) under ``DATA_ROOT/scas_traces_dataset`` by default.

Each row has:
  question_id, question, trace, answer, style_id, style_name, source_dataset

``question_id`` is ``{source_dataset}::{id}`` so IDs do not collide across sources.
``style_name`` is the teacher name (oracle label for analysis; AE training does not use it).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

DEFAULT_TEACHERS = (
    "gemma-4-31b-it",
    "gpt-5-chat_2025-10-03",
    "gpt-oss-120b",
    "llama-3.3-70b-instruct",
    "olmo-3.1-32b-instruct",
    "phi-4-reasoning-plus",
    "qwen2.5-72b-instruct",
    "qwen3-32b",
    "qwen3.5-27b",
)

HF_DATASET = "Student-Centric-Answer-Sampling/scas_verified_teacher_pool"
HF_URL = f"https://huggingface.co/datasets/{HF_DATASET}"


def teacher_style_map(teachers: tuple[str, ...]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(teachers)}


def question_key(source_dataset: str, qid: str) -> str:
    return f"{source_dataset}::{qid}"


def validation_question_ids(all_qids: list[str], validation_fraction: float) -> set[str]:
    if len(all_qids) < 2 or validation_fraction <= 0:
        return set()
    n_val = max(1, int(round(len(all_qids) * validation_fraction)))
    n_val = min(n_val, len(all_qids) - 1)
    return set(all_qids[-n_val:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", type=str, default=HF_DATASET)
    parser.add_argument("--config", type=str, default="all", choices=("all", "hendrycks_math", "deepscaler"))
    parser.add_argument("--teachers", type=str, nargs="+", default=list(DEFAULT_TEACHERS))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="HF DatasetDict path (default: $DATA_ROOT/scas_traces_dataset or ./scas_traces_dataset)",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Optional metadata JSON (default: configs/scas/scas_traces_dataset.json)",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Debug: cap unique questions")
    args = parser.parse_args()

    teachers = tuple(args.teachers)
    style_map = teacher_style_map(teachers)
    allowed = set(teachers)

    out_dir = args.output_dir
    if out_dir is None:
        import os

        data_root = Path(os.environ.get("DATA_ROOT", "."))
        out_dir = data_root / "scas_traces_dataset"

    meta_json = args.meta_json
    if meta_json is None:
        meta_json = Path("configs/scas/scas_traces_dataset.json")

    print(f"Loading {args.hf_dataset} config={args.config!r} …", flush=True)
    ds = load_dataset(args.hf_dataset, args.config, split="train")

    # Pass 1: unique question keys (sorted for a stable val split)
    seen: set[str] = set()
    ordered: list[str] = []
    for row in tqdm(ds, desc="pass1 question ids"):
        teacher = str(row["teacher_name"])
        if teacher not in allowed:
            continue
        key = question_key(str(row.get("source_dataset") or ""), str(row["id"]))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
        if args.max_questions is not None and len(ordered) >= args.max_questions:
            break
    ordered.sort()
    val_qids = validation_question_ids(ordered, args.validation_fraction)
    allowed_qids = set(ordered)
    print(f"Questions: {len(ordered)} (validation={len(val_qids)})", flush=True)

    # Pass 2: rows
    buckets: dict[str, list[dict]] = {"train": [], "validation": []}
    teacher_counts: dict[str, Counter] = {"train": Counter(), "validation": Counter()}

    for row in tqdm(ds, desc="pass2 build traces"):
        teacher = str(row["teacher_name"])
        if teacher not in allowed:
            continue
        source = str(row.get("source_dataset") or "")
        qid_raw = str(row["id"])
        key = question_key(source, qid_raw)
        if key not in allowed_qids:
            continue
        question = str(row.get("instruction") or "").strip()
        trace = str(row.get("teacher_output") or "").strip()
        answer = str(row.get("reference_answer") or "").strip()
        if not question or not trace:
            continue
        split = "validation" if key in val_qids else "train"
        buckets[split].append(
            {
                "question_id": key,
                "question": question,
                "trace": trace,
                "answer": answer,
                "style_id": int(style_map[teacher]),
                "style_name": teacher,
                "source_dataset": source,
            }
        )
        teacher_counts[split][teacher] += 1

    out = DatasetDict(
        {
            "train": Dataset.from_list(buckets["train"]),
            "validation": Dataset.from_list(buckets["validation"]),
        }
    )
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_dir))
    print(
        f"Wrote {out_dir}  train={len(out['train'])}  validation={len(out['validation'])}",
        flush=True,
    )

    meta = {
        "source_hf_dataset": args.hf_dataset,
        "hf_config": args.config,
        "hf_url": HF_URL if args.hf_dataset == HF_DATASET else None,
        "teachers": list(teachers),
        "output_dir": str(out_dir),
        "question_id": "{source_dataset}::{id}",
        "splits": {
            split: {
                "rows": len(buckets[split]),
                "questions": len({r["question_id"] for r in buckets[split]}),
                "teacher_counts": dict(teacher_counts[split]),
            }
            for split in ("train", "validation")
        },
    }
    meta_json = Path(meta_json)
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_json}", flush=True)


if __name__ == "__main__":
    main()
