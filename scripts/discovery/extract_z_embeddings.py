#!/usr/bin/env python3
"""
Extract z(s) (and c(s)) for a contiguous shard of the autoencoder train split.

Write shard NPZ files for parallel multi-GPU extraction, then merge with
merge_z_embedding_shards.py before running clustering_gmm.py --gmm-only.

Usage:
  python scripts/discovery/extract_z_embeddings.py \\
      --checkpoint checkpoints/.../last.ckpt \\
      --hf-dataset scas_traces_dataset \\
      --output-dir data/scas/cluster_gmm_assignments-qwen3-4b \\
      --shard-id 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discovery.clustering_gmm import enrich_records  # noqa: E402
from scripts.eval.eval_gemini_styles import (  # noqa: E402
    extract_embeddings,
    load_gemini_with_styles,
)
from scripts.repo_paths import CONFIG_SCAS  # noqa: E402


def shard_range(n: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    start = (n * shard_id) // num_shards
    end = (n * (shard_id + 1)) // num_shards
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract z(s) shard for parallel embedding")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf-dataset", type=str, default="scas_traces_dataset")
    parser.add_argument("--raw-json", type=str, default=str(CONFIG_SCAS / "scas_traces_dataset.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    import torch
    from src.module import DisentangledLightningModule

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "embedding_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{args.shard_id:04d}_of_{args.num_shards:04d}.npz"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading autoencoder {args.split} split from {args.hf_dataset}...")
    records = enrich_records(load_gemini_with_styles(args.hf_dataset, args.raw_json, split=args.split))
    if args.max_records > 0 and len(records) > args.max_records:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(records), size=args.max_records, replace=False)
        records = [records[int(i)] for i in np.sort(pick)]
        print(f"  Subsampled to {len(records)} records (max_records={args.max_records})")

    start, end = shard_range(len(records), args.shard_id, args.num_shards)
    shard_records = records[start:end]
    print(
        f"Shard {args.shard_id}/{args.num_shards}: rows [{start}:{end}) "
        f"= {len(shard_records)} / {len(records)}"
    )
    if not shard_records:
        raise SystemExit("Empty shard")

    if shard_path.exists():
        print(f"Shard already exists, skipping: {shard_path}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    # Load on CPU then move (cuda map_location triggered mid-extract fatal abort).
    module = DisentangledLightningModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=False,
    )
    module.eval()
    max_len = module.hparams.get("max_len", 4096)

    print(f"Extracting embeddings on {device} (batch_size={args.batch_size})...")
    c_s, z_s = extract_embeddings(
        module, shard_records, args.batch_size, device, max_len=max_len,
    )

    np.savez(
        shard_path,
        c_s=c_s,
        z_s=z_s,
        row_start=np.array(start, dtype=np.int64),
        row_end=np.array(end, dtype=np.int64),
        shard_id=np.array(args.shard_id, dtype=np.int64),
        num_shards=np.array(args.num_shards, dtype=np.int64),
        question_id=np.array([r["question_id"] for r in shard_records]),
        style_id=np.array([r.get("style_id", -1) for r in shard_records], dtype=np.int64),
    )
    print(f"Wrote {shard_path}")

    if args.shard_id == 0:
        manifest = {
            "checkpoint": args.checkpoint,
            "hf_dataset": args.hf_dataset,
            "n_records": len(records),
            "num_shards": args.num_shards,
            "max_records": args.max_records,
            "seed": args.seed,
        }
        manifest_path = args.output_dir / "embedding_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
