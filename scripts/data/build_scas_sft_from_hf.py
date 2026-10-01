#!/usr/bin/env python3
"""
Build vanilla (question-only) SCAS SFT parquet from scas_traces_dataset.

Standard: question-only user prompt (naive mixed-teacher baseline).
ModC:    [teacher_name] prefix + question (oracle teacher conditioning).

Outputs:
  data/scas/standard_full_sft/{train,validation}.parquet
  data/scas/modc_prefix_full_sft/{train,validation}.parquet
  scas_standard_full_sft.json
  scas_modc_prefix_full_sft.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm

from scripts.repo_paths import CONFIG_SCAS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer_utils import configure_tokenizer  # noqa: E402


def row_to_modc_messages(row: dict) -> list[dict]:
    style = str(row["style_name"])
    return [
        {"role": "user", "content": f"[{style}]\n{row['question']}"},
        {"role": "assistant", "content": row["trace"]},
    ]


def row_to_standard_messages(row: dict) -> list[dict]:
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["trace"]},
    ]


def hf_rows_to_df(rows: list[dict], standard: bool) -> pd.DataFrame:
    out = []
    msg_fn = row_to_standard_messages if standard else row_to_modc_messages
    for row in tqdm(rows, desc="standard" if standard else "modc"):
        out.append(
            {
                "messages": json.dumps(msg_fn(row)),
                "question_id": row["question_id"],
                "style_id": int(row["style_id"]),
                "style_name": str(row["style_name"]),
                "answer": str(row.get("answer", "")),
                "source_dataset": str(row.get("source_dataset", "")),
            }
        )
    return pd.DataFrame(out)


def verify_chat_lengths(df: pd.DataFrame, tokenizer, max_tokens: int) -> dict:
    over = 0
    lengths = []
    for messages in tqdm(df["messages"], desc="verify chat tokens"):
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n)
        if n > max_tokens:
            over += 1
    return {
        "n": len(df),
        "over_max_chat_tokens": over,
        "frac_over": over / len(df) if len(df) else 0.0,
        "p50": float(sorted(lengths)[len(lengths) // 2]) if lengths else 0,
        "p99": float(sorted(lengths)[int(0.99 * (len(lengths) - 1))]) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SCAS standard + ModC-prefix SFT parquet")
    parser.add_argument(
        "--hf-dataset",
        type=Path,
        default=Path("scas_traces_dataset"),
    )
    parser.add_argument(
        "--modc-output-dir",
        type=Path,
        default=Path("data/scas/modc_prefix_full_sft"),
    )
    parser.add_argument(
        "--standard-output-dir",
        type=Path,
        default=Path("data/scas/standard_full_sft"),
    )
    parser.add_argument("--verify-chat-tokens", type=int, default=4096)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument(
        "--modc-json-out",
        type=Path,
        default=CONFIG_SCAS / "scas_modc_prefix_full_sft.json",
    )
    parser.add_argument(
        "--standard-json-out",
        type=Path,
        default=CONFIG_SCAS / "scas_standard_full_sft.json",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation"],
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Only write question-only standard SFT parquet",
    )
    args = parser.parse_args()

    print(f"Loading {args.hf_dataset}...", flush=True)
    ds = load_from_disk(str(args.hf_dataset))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(ds)!r}")

    args.standard_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.standard_only:
        args.modc_output_dir.mkdir(parents=True, exist_ok=True)

    modc_stats: dict = {}
    std_stats: dict = {}
    tokenizer = None
    if args.verify_chat_tokens > 0:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {args.tokenizer}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        tokenizer = configure_tokenizer(args.tokenizer, tokenizer)

    for split in args.splits:
        if split not in ds:
            raise SystemExit(f"Split {split!r} not in {args.hf_dataset}")
        rows = [dict(r) for r in ds[split]]
        print(f"\n{split}: {len(rows)} rows", flush=True)

        std_df = hf_rows_to_df(rows, standard=True)
        std_path = args.standard_output_dir / f"{split}.parquet"
        std_df.to_parquet(std_path, index=False)
        print(f"  standard -> {std_path}")

        split_info = {
            "rows": len(std_df),
            "style_counts": dict(Counter(std_df["style_name"])),
        }
        if tokenizer is not None:
            split_info["standard_chat_token_check"] = verify_chat_lengths(
                std_df, tokenizer, args.verify_chat_tokens,
            )
        std_stats[split] = split_info

        if not args.standard_only:
            modc_df = hf_rows_to_df(rows, standard=False)
            modc_path = args.modc_output_dir / f"{split}.parquet"
            modc_df.to_parquet(modc_path, index=False)
            print(f"  modc     -> {modc_path}")
            modc_split_info = dict(split_info)
            if tokenizer is not None:
                modc_split_info["modc_chat_token_check"] = verify_chat_lengths(
                    modc_df, tokenizer, args.verify_chat_tokens,
                )
            modc_stats[split] = modc_split_info

    std_json = {
        "source_hf_dataset": str(args.hf_dataset),
        "hf_url": "https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool",
        "prompt_format": "question_only (no teacher prefix)",
        "splits": std_stats,
    }
    args.standard_json_out.write_text(json.dumps(std_json, indent=2))
    print(json.dumps(std_json, indent=2))
    print(f"Wrote {args.standard_json_out}")

    if not args.standard_only:
        modc_json = {
            "source_hf_dataset": str(args.hf_dataset),
            "hf_url": "https://huggingface.co/datasets/Student-Centric-Answer-Sampling/scas_verified_teacher_pool",
            "prompt_format": "[teacher_name] prefix + question",
            "splits": modc_stats,
        }
        args.modc_json_out.write_text(json.dumps(modc_json, indent=2))
        print(f"Wrote {args.modc_json_out}")


if __name__ == "__main__":
    main()
