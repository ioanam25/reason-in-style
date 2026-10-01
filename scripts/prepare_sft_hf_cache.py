#!/usr/bin/env python3
"""Build tokenized HuggingFace on-disk cache from SFT parquet (once, before DDP training).

Writes ``{parquet_dir}/.hf_tokenized/{train,validation}/``.

Usage:
  python scripts/prepare_sft_hf_cache.py data/scas/standard_full_sft \\
      --tokenizer Qwen/Qwen3-0.6B-Base --max-seq-length 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mod_c.train import configure_tokenizer_and_response_template  # noqa: E402


def parse_messages(raw) -> list[dict]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()
    raw[-1]["content"] = raw[-1]["content"].strip()
    return raw


def tokenize_split(
    parquet_path: Path,
    cache_path: Path,
    tokenizer,
    max_seq_length: int,
    num_proc: int,
    force: bool,
) -> None:
    marker = cache_path / "dataset_info.json"
    if marker.is_file() and not force:
        from datasets import load_from_disk

        ds = load_from_disk(str(cache_path))
        if "input_ids" in ds.column_names:
            print(f"  skip {cache_path} ({len(ds)} rows, tokenized)", flush=True)
            return

    print(f"  {parquet_path} -> {cache_path}", flush=True)
    dataset = Dataset(pq.ParquetFile(parquet_path).read())

    def tokenize_example(example: dict) -> dict:
        msgs = parse_messages(example["messages"])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc=f"tokenize {parquet_path.name}",
        num_proc=num_proc,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    print(f"  wrote {len(dataset)} rows to {cache_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_dir", type=Path)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer, _ = configure_tokenizer_and_response_template(args.tokenizer, tokenizer)

    cache_root = args.parquet_dir / ".hf_tokenized"
    for split in ("train", "validation"):
        parquet_path = args.parquet_dir / f"{split}.parquet"
        if not parquet_path.is_file():
            raise SystemExit(f"Missing {parquet_path}")
        tokenize_split(
            parquet_path,
            cache_root / split,
            tokenizer,
            args.max_seq_length,
            args.num_proc,
            args.force,
        )


if __name__ == "__main__":
    main()
