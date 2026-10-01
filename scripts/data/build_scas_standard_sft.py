#!/usr/bin/env python3
"""
Build vanilla (question-only) SCAS SFT parquet.

Two-pass stream from HuggingFace — never holds all traces in memory.

Source:
  https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool

Outputs:
  data/scas/standard_full_sft/{train,validation}.parquet
  data/scas/modc_prefix_full_sft/{train,validation}.parquet  (--also-modc or --modc-only)
  scas_standard_full_sft.json
  scas_modc_prefix_full_sft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm.auto import tqdm

from scripts.repo_paths import CONFIG_SCAS
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_utils import configure_tokenizer  # noqa: E402

HF_DATASET = "Student-Centric-Answer-Sampling/scas_verified_teacher_pool"
HF_URL = "https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool"

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

SCHEMA = pa.schema(
    [
        ("messages", pa.string()),
        ("question_id", pa.string()),
        ("style_id", pa.int64()),
        ("style_name", pa.string()),
        ("answer", pa.string()),
        ("source_dataset", pa.string()),
    ]
)


def teacher_style_map(teachers: tuple[str, ...]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(teachers)}


def validation_question_ids(all_qids: list[str], validation_fraction: float) -> set[str]:
    if len(all_qids) < 2 or validation_fraction <= 0:
        return set(all_qids)
    n_val = max(1, int(round(len(all_qids) * validation_fraction)))
    n_val = min(n_val, len(all_qids) - 1)
    return set(all_qids[-n_val:])


def question_key(source_dataset: str, qid: str) -> str:
    return f"{source_dataset}::{qid}"


def collect_question_ids(config: str, teachers: set[str], max_rows: int | None) -> list[str]:
    ds = load_dataset(HF_DATASET, config, split="train")
    seen: set[str] = set()
    ordered: list[str] = []
    for row in tqdm(ds, desc="pass1 question ids"):
        if str(row["teacher_name"]) not in teachers:
            continue
        key = question_key(str(row.get("source_dataset") or ""), str(row["id"]))
        if key in seen:
            continue
        if max_rows is not None and len(seen) >= max_rows:
            continue
        seen.add(key)
        ordered.append(key)
    ordered.sort()
    return ordered


def user_content(question: str, teacher: str | None) -> str:
    if teacher is None:
        return question
    return f"[{teacher}]\n{question}"


def stream_to_parquet(
    config: str,
    teachers: tuple[str, ...],
    style_map: dict[str, int],
    val_qids: set[str],
    standard_output_dir: Path | None,
    modc_output_dir: Path | None,
    allowed_qids: set[str] | None,
) -> dict[str, dict[str, Counter]]:
    if standard_output_dir is None and modc_output_dir is None:
        raise ValueError("At least one of standard_output_dir or modc_output_dir is required")

    allowed = set(teachers)
    ds = load_dataset(HF_DATASET, config, split="train")

    def setup_variant(output_dir: Path | None) -> dict | None:
        if output_dir is None:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "output_dir": output_dir,
            "paths": {
                "train": output_dir / "train.parquet",
                "validation": output_dir / "validation.parquet",
            },
            "writers": {"train": None, "validation": None},
            "counts": {"train": Counter(), "validation": Counter()},
            "row_counts": {"train": 0, "validation": 0},
            "batch": {k: {col: [] for col in SCHEMA.names} for k in ("train", "validation")},
        }

    standard = setup_variant(standard_output_dir)
    modc = setup_variant(modc_output_dir)
    chunk = 2000

    def flush(variant: dict, split: str) -> None:
        batch = variant["batch"][split]
        if not batch["messages"]:
            return
        table = pa.table(batch, schema=SCHEMA)
        writers = variant["writers"]
        paths = variant["paths"]
        if writers[split] is None:
            writers[split] = pq.ParquetWriter(paths[split], SCHEMA)
        writers[split].write_table(table)
        for col in batch:
            batch[col].clear()

    def append_row(variant: dict, split: str, teacher: str | None, row_fields: dict) -> None:
        question = row_fields["question"]
        trace = row_fields["trace"]
        messages = json.dumps(
            [
                {"role": "user", "content": user_content(question, teacher)},
                {"role": "assistant", "content": trace},
            ]
        )
        batch = variant["batch"][split]
        batch["messages"].append(messages)
        batch["question_id"].append(row_fields["qid"])
        batch["style_id"].append(row_fields["style_id"])
        batch["style_name"].append(row_fields["teacher"])
        batch["answer"].append(row_fields["answer"])
        batch["source_dataset"].append(row_fields["source_dataset"])
        variant["counts"][split][row_fields["teacher"]] += 1
        variant["row_counts"][split] += 1
        if len(batch["messages"]) >= chunk:
            flush(variant, split)

    for row in tqdm(ds, desc="pass2 write parquet"):
        teacher = str(row["teacher_name"])
        if teacher not in allowed:
            continue
        qid = str(row["id"])
        key = question_key(str(row.get("source_dataset") or ""), qid)
        if allowed_qids is not None and key not in allowed_qids:
            continue

        question = str(row.get("instruction") or "").strip()
        trace = str(row.get("teacher_output") or "").strip()
        answer = str(row.get("reference_answer") or "").strip()
        if not question or not trace:
            continue

        split = "validation" if key in val_qids else "train"
        row_fields = {
            "qid": qid,
            "question": question,
            "trace": trace,
            "answer": answer,
            "style_id": style_map[teacher],
            "teacher": teacher,
            "source_dataset": str(row.get("source_dataset") or ""),
        }
        if standard is not None:
            append_row(standard, split, None, row_fields)
        if modc is not None:
            append_row(modc, split, teacher, row_fields)

    results: dict[str, dict[str, Counter]] = {}
    for name, variant in (("standard", standard), ("modc", modc)):
        if variant is None:
            continue
        for split in ("train", "validation"):
            flush(variant, split)
            if variant["writers"][split] is not None:
                variant["writers"][split].close()
        print(
            f"Wrote {name} train={variant['row_counts']['train']} "
            f"validation={variant['row_counts']['validation']}",
            flush=True,
        )
        results[name] = variant["counts"]
    return results


def verify_parquet_lengths(path: Path, tokenizer, max_tokens: int) -> dict:
    table = pq.read_table(path, columns=["messages"])
    over = 0
    lengths = []
    for messages in tqdm(table.column("messages").to_pylist(), desc=f"verify {path.name}"):
        messages = json.loads(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n > max_tokens:
            over += 1
    lengths.sort()
    return {
        "n": len(lengths),
        "over_max_chat_tokens": over,
        "frac_over": over / len(lengths) if lengths else 0.0,
        "p50": float(lengths[len(lengths) // 2]) if lengths else 0,
        "p99": float(lengths[int(0.99 * (len(lengths) - 1))]) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SCAS SFT parquet from HuggingFace")
    parser.add_argument("--config", type=str, default="all", choices=("all", "hendrycks_math", "deepscaler"))
    parser.add_argument("--teachers", type=str, nargs="+", default=list(DEFAULT_TEACHERS))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit questions (debug)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/scas/standard_full_sft"))
    parser.add_argument("--modc-output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=CONFIG_SCAS / "scas_standard_full_sft.json")
    parser.add_argument("--modc-json-out", type=Path, default=CONFIG_SCAS / "scas_modc_prefix_full_sft.json")
    parser.add_argument("--also-modc", action="store_true", help="Write standard + ModC parquet in one HF pass")
    parser.add_argument("--modc-only", action="store_true", help="Write ModC parquet only")
    parser.add_argument("--verify-chat-tokens", type=int, default=4096)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B-Base")
    args = parser.parse_args()

    if args.modc_only:
        standard_dir = None
        modc_dir = args.modc_output_dir or Path("data/scas/modc_prefix_full_sft")
    elif args.also_modc:
        standard_dir = args.output_dir
        modc_dir = args.modc_output_dir or Path("data/scas/modc_prefix_full_sft")
    else:
        standard_dir = args.output_dir
        modc_dir = None

    teachers = tuple(args.teachers)
    teacher_set = set(teachers)
    style_map = teacher_style_map(teachers)

    print(f"Loading HF dataset: {HF_DATASET} config={args.config!r}", flush=True)
    qids = collect_question_ids(args.config, teacher_set, args.max_rows)
    val_qids = validation_question_ids(qids, args.validation_fraction)
    print(f"Questions: {len(qids)} (validation={len(val_qids)})", flush=True)

    counts = stream_to_parquet(
        config=args.config,
        teachers=teachers,
        style_map=style_map,
        val_qids=val_qids,
        standard_output_dir=standard_dir,
        modc_output_dir=modc_dir,
        allowed_qids=set(qids) if args.max_rows is not None else None,
    )

    if standard_dir is not None and "standard" in counts:
        train_path = standard_dir / "train.parquet"
        val_path = standard_dir / "validation.parquet"
        splits = {
            "train": {
                "rows": int(pq.read_metadata(train_path).num_rows),
                "questions": len(qids) - len(val_qids),
                "style_counts": dict(counts["standard"]["train"]),
            },
            "validation": {
                "rows": int(pq.read_metadata(val_path).num_rows),
                "questions": len(val_qids),
                "style_counts": dict(counts["standard"]["validation"]),
            },
        }
        if args.verify_chat_tokens > 0:
            print(f"Loading tokenizer: {args.tokenizer}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            tokenizer = configure_tokenizer(args.tokenizer, tokenizer)
            splits["train"]["standard_chat_token_check"] = verify_parquet_lengths(
                train_path, tokenizer, args.verify_chat_tokens,
            )
            splits["validation"]["standard_chat_token_check"] = verify_parquet_lengths(
                val_path, tokenizer, args.verify_chat_tokens,
            )
        payload = {
            "source_hf_dataset": HF_DATASET,
            "hf_config": args.config,
            "hf_url": HF_URL,
            "teachers": list(teachers),
            "prompt_format": "question_only (no teacher prefix)",
            "output_dir": str(standard_dir),
            "splits": splits,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(f"Wrote {train_path}")
        print(f"Wrote {val_path}")
        print(f"Wrote {args.json_out}")

    if modc_dir is not None and "modc" in counts:
        train_path = modc_dir / "train.parquet"
        val_path = modc_dir / "validation.parquet"
        splits = {
            "train": {
                "rows": int(pq.read_metadata(train_path).num_rows),
                "questions": len(qids) - len(val_qids),
                "style_counts": dict(counts["modc"]["train"]),
            },
            "validation": {
                "rows": int(pq.read_metadata(val_path).num_rows),
                "questions": len(val_qids),
                "style_counts": dict(counts["modc"]["validation"]),
            },
        }
        if args.verify_chat_tokens > 0:
            print(f"Loading tokenizer: {args.tokenizer}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            tokenizer = configure_tokenizer(args.tokenizer, tokenizer)
            splits["train"]["modc_chat_token_check"] = verify_parquet_lengths(
                train_path, tokenizer, args.verify_chat_tokens,
            )
            splits["validation"]["modc_chat_token_check"] = verify_parquet_lengths(
                val_path, tokenizer, args.verify_chat_tokens,
            )
        payload = {
            "source_hf_dataset": HF_DATASET,
            "hf_config": args.config,
            "hf_url": HF_URL,
            "teachers": list(teachers),
            "prompt_format": "[teacher_name] prefix + question",
            "output_dir": str(modc_dir),
            "splits": splits,
        }
        args.modc_json_out.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(f"Wrote {train_path}")
        print(f"Wrote {val_path}")
        print(f"Wrote {args.modc_json_out}")


if __name__ == "__main__":
    main()
